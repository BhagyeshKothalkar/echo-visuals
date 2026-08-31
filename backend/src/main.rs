use agentic_prompt_improver::{
    agent::Agent,
    config::Config,
    domain::FeedbackGrade,
    harness::Harness,
    llm::RigOpenAiProvider,
    ports::ToolRegistry,
    qdrant::{QdrantPromptRepository, QwenEmbeddingEmbedder, SearchSkillsTool},
    redis_store::{GetFeedbackTool, RecordFeedbackTool, RedisFeedbackStore},
};
use base64::Engine;
use clap::{Parser, Subcommand};
use std::{path::PathBuf, sync::Arc};

fn image_reference(value: &str) -> anyhow::Result<String> {
    if value.starts_with("data:") || value.starts_with("http://") || value.starts_with("https://") {
        return Ok(value.into());
    }
    let bytes = std::fs::read(value)?;
    let mime = match std::path::Path::new(value)
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("png")
    {
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "gif" => "image/gif",
        _ => "image/png",
    };
    Ok(format!(
        "data:{mime};base64,{}",
        base64::engine::general_purpose::STANDARD.encode(bytes)
    ))
}

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
        #[arg(long)]
        image: Option<String>,
    },
    Interactive {
        #[arg(long)]
        target: String,
        #[arg(long)]
        image: Option<String>,
    },
}
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    tracing_subscriber::fmt().with_env_filter("info").init();
    let config = Config::from_sources(Some(&cli.config))?;
    let embedder = Arc::new(QwenEmbeddingEmbedder::new(&config.embedding));
    let qdrant = Arc::new(QdrantPromptRepository::new(&config.qdrant, embedder));
    if matches!(cli.command, Command::Init) {
        qdrant.initialize().await?;
        println!("initialized {}", config.qdrant.collection);
        return Ok(());
    }
    qdrant.initialize().await?;
    let redis = Arc::new(RedisFeedbackStore::new(&config.redis).await?);
    let llm = Arc::new(RigOpenAiProvider::new(
        config.analyst.clone(),
        config.optimizer.clone(),
        config.curator.clone(),
    )?);
    let tools = Arc::new(ToolRegistry::new(vec![
        Arc::new(SearchSkillsTool(qdrant.clone())),
        Arc::new(GetFeedbackTool(redis.clone())),
        Arc::new(RecordFeedbackTool(redis)),
    ])?);
    let image = match &cli.command {
        Command::Run { image, .. } | Command::Interactive { image, .. } => image
            .as_deref()
            .map(image_reference)
            .transpose()?
            .unwrap_or_default(),
        Command::Init => unreachable!(),
    };
    let agent = Agent::new(
        match &cli.command {
            Command::Run { target, .. } | Command::Interactive { target, .. } => target.clone(),
            Command::Init => unreachable!(),
        },
        image,
        tools,
        qdrant,
        llm,
    );
    let harness = Harness::new();
    match cli.command {
        Command::Run { .. } => {
            let c = harness.run_iteration(&agent).await?;
            println!("id={}\n{}", c.id, c.text);
        }
        Command::Interactive { .. } => loop {
            let c = harness.run_iteration(&agent).await?;
            println!("id={}\n{}\nGrade [positive/negative/q]:", c.id, c.text);
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
