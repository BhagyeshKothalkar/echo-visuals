use crate::{
    config::LlmConfig,
    ports::{LlmError, LlmProvider},
    prompts::{PromptAssets, PromptContext, RenderedPrompt},
};
use async_trait::async_trait;
use rig_core::{
    client::CompletionClient,
    completion::{AssistantContent, CompletionModel},
    providers::openai,
};
pub struct RigOpenAiProvider {
    client: openai::CompletionsClient,
    model: String,
    system: String,
    temperature: f64,
    max_tokens: u64,
}
impl RigOpenAiProvider {
    pub fn new(config: LlmConfig, assets: PromptAssets) -> Result<Self, LlmError> {
        let key = config
            .api_key
            .ok_or_else(|| LlmError::Request(anyhow::anyhow!("LLM_API_KEY is required")))?;
        let client = openai::CompletionsClient::builder()
            .api_key(key)
            .base_url(config.base_url)
            .build()
            .map_err(|e| LlmError::Request(anyhow::Error::msg(e.to_string())))?;
        let system = assets
            .render(&PromptContext {
                target: String::new(),
                positive: vec![],
                negative: vec![],
                discoveries: vec![],
            })
            .system;
        Ok(Self {
            client,
            model: config.model,
            system,
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
        _: &PromptContext,
    ) -> Result<String, LlmError> {
        let model = self.client.completion_model(&self.model);
        let request = model
            .completion_request(&prompt.user)
            .preamble(self.system.clone())
            .temperature(self.temperature)
            .max_tokens(self.max_tokens)
            .build();
        let response = model
            .completion(request)
            .await
            .map_err(|e| LlmError::Request(anyhow::Error::msg(e.to_string())))?;
        let candidate = response
            .choice
            .into_iter()
            .find_map(|item| match item {
                AssistantContent::Text(text) => Some(text.text),
                _ => None,
            })
            .unwrap_or_default()
            .trim()
            .to_string();
        if candidate.is_empty() {
            Err(LlmError::EmptyCandidate)
        } else {
            Ok(candidate)
        }
    }
}
