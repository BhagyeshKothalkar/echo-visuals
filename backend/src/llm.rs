use crate::{
    config::LlmConfig,
    ports::{GenerationResult, LlmError, LlmProvider, ToolCallStats, ToolRegistry},
    prompts::RenderedPrompt,
};
use async_trait::async_trait;
use rig_agent::completion::Prompt;
use rig_core::{
    client::CompletionClient,
    providers::openai,
    tool::{PortableDynamicTool, ToolExecutionError, ToolOutput},
};
use std::sync::{Arc, Mutex};
pub struct RigOpenAiProvider {
    client: openai::CompletionsClient,
    model: String,
    temperature: f64,
    max_tokens: u64,
}
impl RigOpenAiProvider {
    pub fn new(config: LlmConfig) -> Result<Self, LlmError> {
        let key = config
            .api_key
            .ok_or_else(|| LlmError::Request(anyhow::anyhow!("LLM_API_KEY is required")))?;
        let client = openai::CompletionsClient::builder()
            .api_key(key)
            .base_url(config.base_url)
            .build()
            .map_err(|e| LlmError::Request(anyhow::Error::msg(e.to_string())))?;
        Ok(Self {
            client,
            model: config.model,
            temperature: config.temperature as f64,
            max_tokens: config.max_tokens as u64,
        })
    }
}
#[async_trait]
impl LlmProvider for RigOpenAiProvider {
    async fn generate(
        &self,
        prompt: &RenderedPrompt,
        tools: Arc<ToolRegistry>,
    ) -> Result<GenerationResult, LlmError> {
        let model = self.client.completion_model(&self.model);
        let stats = Arc::new(Mutex::new(ToolCallStats::default()));
        let saved = Arc::new(Mutex::new(None));
        let preamble = prompt.system.clone();
        let agent = rig_agent::AgentBuilder::new(model)
            .preamble(&preamble)
            .temperature(self.temperature)
            .max_tokens(self.max_tokens)
            .default_max_turns(8)
            .portable_dynamic_tool(native_tool("search_skills", "Search stored prompt skills relevant to the target.", serde_json::json!({"type":"object","properties":{"query":{"type":"string"},"limit":{"type":"integer"}},"required":["query","limit"]}), tools.clone(), stats.clone(), saved.clone()))
            .portable_dynamic_tool(native_tool("save_skill", "Persist a useful reusable skill.", serde_json::json!({"type":"object","properties":{"skill_id":{"type":"string"},"name":{"type":"string"},"description":{"type":"string"},"knowledge":{"type":"object"},"usage":{"type":"object"},"retrieval":{"type":"object"},"lifecycle":{"type":"object"}},"required":["skill_id","name","description","knowledge","usage","retrieval","lifecycle"]}), tools.clone(), stats.clone(), saved.clone()))
            .portable_dynamic_tool(native_tool("get_feedback", "Retrieve positive or negative feedback examples on demand.", serde_json::json!({"type":"object","properties":{"grade":{"type":"string","enum":["Positive","Negative"]},"limit":{"type":"integer"}},"required":["grade","limit"]}), tools, stats.clone(), saved.clone()))
            .build();
        let candidate = agent
            .prompt(&prompt.user)
            .await
            .map_err(|e| LlmError::Request(anyhow::Error::msg(e.to_string())))?
            .trim()
            .to_string();
        if candidate.is_empty() {
            Err(LlmError::EmptyCandidate)
        } else {
            Ok(GenerationResult {
                text: candidate,
                stats: stats.lock().unwrap().clone(),
                saved_skill: saved.lock().unwrap().clone(),
            })
        }
    }
}

fn native_tool(
    name: &'static str,
    description: &'static str,
    parameters: serde_json::Value,
    registry: Arc<ToolRegistry>,
    stats: Arc<Mutex<ToolCallStats>>,
    saved: Arc<Mutex<Option<crate::domain::SkillRecord>>>,
) -> PortableDynamicTool {
    PortableDynamicTool::new(name, description, parameters, move |input| {
        let registry = registry.clone();
        let stats = stats.clone();
        let saved = saved.clone();
        Box::pin(async move {
            let result = registry
                .call_with_stats(name, input, &stats)
                .await
                .map_err(|e| ToolExecutionError::other(e.to_string()))?;
            if name == "save_skill" {
                let record = serde_json::from_value(result.clone()).map_err(|e| {
                    ToolExecutionError::other(format!("invalid save_skill result: {e}"))
                })?;
                *saved.lock().unwrap() = Some(record);
            }
            Ok(ToolOutput::from(result))
        })
    })
}
