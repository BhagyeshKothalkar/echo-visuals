use crate::{
    config::{LlmConfig, VlmConfig},
    domain::{AnalystInput, AnalystOutput, OptimizerOutput, SkillRecord},
    ports::{LlmError, LlmProvider, ToolRegistry},
};
use async_trait::async_trait;
use rig_agent::completion::Prompt;
use rig_core::{
    client::CompletionClient,
    providers::openai,
    tool::{PortableDynamicTool, ToolExecutionError, ToolOutput},
};
use std::sync::Arc;

pub struct RigOpenAiProvider {
    analyst: RoleClient,
    optimizer: RoleClient,
    curator: RoleClient,
    analyst_turns: usize,
}
struct RoleClient {
    client: openai::CompletionsClient,
    model: String,
    temperature: f64,
    max_tokens: u64,
}
impl RigOpenAiProvider {
    pub fn new(
        analyst: VlmConfig,
        optimizer: LlmConfig,
        curator: LlmConfig,
    ) -> Result<Self, LlmError> {
        let analyst_turns = analyst.max_turns.clamp(1, 4);
        Ok(Self {
            analyst: RoleClient::new(
                analyst.base_url,
                analyst.model,
                analyst.temperature,
                analyst.max_tokens,
                analyst.api_key,
            )?,
            optimizer: RoleClient::from_llm(optimizer)?,
            curator: RoleClient::from_llm(curator)?,
            analyst_turns,
        })
    }
}
impl RoleClient {
    fn new(
        base_url: String,
        model: String,
        temperature: f32,
        max_tokens: u32,
        api_key: Option<String>,
    ) -> Result<Self, LlmError> {
        let key =
            api_key.ok_or_else(|| LlmError::Request(anyhow::anyhow!("LLM_API_KEY is required")))?;
        let client = openai::CompletionsClient::builder()
            .api_key(key)
            .base_url(base_url)
            .build()
            .map_err(|e| LlmError::Request(anyhow::Error::msg(e.to_string())))?;
        Ok(Self {
            client,
            model,
            temperature: temperature as f64,
            max_tokens: max_tokens as u64,
        })
    }
    fn from_llm(config: LlmConfig) -> Result<Self, LlmError> {
        Self::new(
            config.base_url,
            config.model,
            config.temperature,
            config.max_tokens,
            config.api_key,
        )
    }
    async fn complete(
        &self,
        system: &str,
        user: impl Into<rig_core::completion::Message> + Send,
    ) -> Result<String, LlmError> {
        let text = rig_agent::AgentBuilder::new(self.client.completion_model(&self.model))
            .preamble(system)
            .temperature(self.temperature)
            .max_tokens(self.max_tokens)
            .default_max_turns(1)
            .build()
            .prompt(user)
            .await
            .map_err(|e| LlmError::Request(anyhow::Error::msg(e.to_string())))?
            .trim()
            .to_string();
        if text.is_empty() {
            Err(LlmError::EmptyCandidate)
        } else {
            Ok(text)
        }
    }
}
#[async_trait]
impl LlmProvider for RigOpenAiProvider {
    async fn analyze(
        &self,
        input: &AnalystInput,
        tools: Arc<ToolRegistry>,
    ) -> Result<AnalystOutput, LlmError> {
        let search = native_tool(
            "search_skills",
            "Find relevant reusable skills.",
            serde_json::json!({"type":"object","properties":{"query":{"type":"string"},"limit":{"type":"integer","maximum":8}},"required":["query","limit"]}),
            tools.clone(),
        );
        let feedback = native_tool(
            "get_feedback",
            "Retrieve prior examples.",
            serde_json::json!({"type":"object","properties":{"grade":{"type":"string","enum":["Positive","Negative"]},"limit":{"type":"integer","maximum":8}},"required":["grade","limit"]}),
            tools,
        );
        use rig_core::completion::message::{DocumentSourceKind, Image, Text, UserContent};
        let image = UserContent::Image(Image {
            data: DocumentSourceKind::Url(input.image.clone()),
            ..Default::default()
        });
        let prompt = rig_core::completion::Message::from(vec![
            UserContent::Text(Text::new(&input.target)),
            image,
        ]);
        let text =
            rig_agent::AgentBuilder::new(self.analyst.client.completion_model(&self.analyst.model))
                .preamble(include_str!("../prompts/agents/analyst-system.md"))
                .temperature(self.analyst.temperature)
                .max_tokens(self.analyst.max_tokens)
                .default_max_turns(self.analyst_turns)
                .portable_dynamic_tool(search)
                .portable_dynamic_tool(feedback)
                .build()
                .prompt(prompt)
                .await
                .map_err(|e| LlmError::Request(anyhow::Error::msg(e.to_string())))?
                .trim()
                .to_string();
        parse("analyst", &text)
    }
    async fn optimize(&self, input: &str) -> Result<OptimizerOutput, LlmError> {
        parse(
            "optimizer",
            &self
                .optimizer
                .complete(include_str!("../prompts/agents/optimizer-system.md"), input)
                .await?,
        )
    }
    async fn curate(&self, input: &str) -> Result<SkillRecord, LlmError> {
        parse(
            "curator",
            &self
                .curator
                .complete(include_str!("../prompts/agents/curator-system.md"), input)
                .await?,
        )
    }
}
fn parse<T: serde::de::DeserializeOwned>(role: &str, text: &str) -> Result<T, LlmError> {
    serde_json::from_str(text).map_err(|source| LlmError::MalformedOutput {
        role: role.into(),
        source,
    })
}
fn native_tool(
    name: &'static str,
    description: &'static str,
    parameters: serde_json::Value,
    registry: Arc<ToolRegistry>,
) -> PortableDynamicTool {
    PortableDynamicTool::new(name, description, parameters, move |input| {
        let registry = registry.clone();
        Box::pin(async move {
            let value = registry
                .call(name, input)
                .await
                .map_err(|e| ToolExecutionError::other(e.to_string()))?;
            Ok(ToolOutput::from(value))
        })
    })
}
