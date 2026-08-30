use agentic_prompt_improver::{
    agent::Agent,
    config::Config,
    domain::FeedbackGrade,
    harness::Harness,
    llm::RigOpenAiProvider,
    ports::ToolRegistry,
    prompts::PromptAssets,
    qdrant::{DiscoverPromptsTool, InsertPromptTool, QdrantPromptRepository},
    redis_store::{RecordFeedbackTool, RedisFeedbackStore, TopFeedbackExamplesTool},
};
use clap::{Parser, Subcommand};
use std::{path::PathBuf, sync::Arc};

#[derive(Parser)]
#[command(name = "prompt-improver")]
struct Cli {
    #[arg(long, default_value = "config.toml")]
    config: PathBuf,
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    Init,
    Run {
        #[arg(long)]
        target: String,
    },
    Interactive {
        #[arg(long)]
        target: String,
    },
}
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    tracing_subscriber::fmt().with_env_filter("info").init();
    let config = Config::from_sources(Some(&cli.config))?;
    let qdrant = Arc::new(QdrantPromptRepository::new(&config.qdrant));
    let redis = Arc::new(RedisFeedbackStore::new(&config.redis).await?);
    if matches!(cli.command, Command::Init) {
        qdrant.initialize().await?;
        println!("initialized {}", config.qdrant.collection);
        return Ok(());
    }
    qdrant.initialize().await?;
    let assets = PromptAssets::load(&config.prompts)?;
    let llm = Arc::new(RigOpenAiProvider::new(config.llm.clone(), assets.clone())?);
    let tools = Arc::new(ToolRegistry::new(vec![
        Arc::new(DiscoverPromptsTool(qdrant.clone())),
        Arc::new(InsertPromptTool(qdrant)),
        Arc::new(TopFeedbackExamplesTool(redis.clone())),
        Arc::new(RecordFeedbackTool(redis)),
    ])?);
    let agent = Agent::new(
        match &cli.command {
            Command::Run { target } | Command::Interactive { target } => target.clone(),
            Command::Init => unreachable!(),
        },
        tools,
        llm,
        assets,
    );
    let harness = Harness::new(config.retrieval);
    match cli.command {
        Command::Run { .. } => {
            let c = harness.run_iteration(&agent).await?;
            println!("id={}\n{}", c.record.id, c.record.text);
        }
        Command::Interactive { .. } => loop {
            let c = harness.run_iteration(&agent).await?;
            println!(
                "id={}\n{}\nGrade [positive/negative/q]:",
                c.record.id, c.record.text
            );
            let mut input = String::new();
            tokio::io::AsyncBufReadExt::read_line(
                &mut tokio::io::BufReader::new(tokio::io::stdin()),
                &mut input,
            )
            .await?;
            match input.trim() {
                "positive" => agent.grade(&c, FeedbackGrade::Positive).await?,
                "negative" => agent.grade(&c, FeedbackGrade::Negative).await?,
                "q" => break,
                _ => println!("enter positive, negative, or q"),
            }
        },
        Command::Init => {}
    }
    Ok(())
}
