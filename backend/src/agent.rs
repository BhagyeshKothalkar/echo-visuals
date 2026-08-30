use crate::{
    config::RetrievalConfig,
    domain::{CandidatePrompt, FeedbackGrade},
    ports::{LlmError, LlmProvider, ToolError, ToolRegistry},
    prompts::{PromptAssets, PromptContext},
};
use std::sync::Arc;
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error(transparent)]
    Tool(#[from] ToolError),
    #[error(transparent)]
    Llm(#[from] LlmError),
    #[error("prompt rendering failed: {0}")]
    Prompt(String),
}
pub struct Agent {
    target: String,
    tools: Arc<ToolRegistry>,
    llm: Arc<dyn LlmProvider>,
    assets: PromptAssets,
}
impl Agent {
    pub fn new(
        target: String,
        tools: Arc<ToolRegistry>,
        llm: Arc<dyn LlmProvider>,
        assets: PromptAssets,
    ) -> Self {
        Self {
            target,
            tools,
            llm,
            assets,
        }
    }
    pub async fn iterate(&self, limits: &RetrievalConfig) -> Result<CandidatePrompt, AgentError> {
        let positive = self
            .top_examples(FeedbackGrade::Positive, limits.positive_limit)
            .await?;
        let negative = self
            .top_examples(FeedbackGrade::Negative, limits.negative_limit)
            .await?;
        let discoveries = self.discover(limits.discovery_limit).await?;
        let context = PromptContext {
            target: self.target.clone(),
            positive,
            negative,
            discoveries,
        };
        let rendered = self.assets.render(&context);
        let text = self.llm.generate(&rendered, &context).await?;
        let record = serde_json::from_value(
            self.tools
                .call("insert_prompt", serde_json::json!({"text": text}))
                .await?,
        )
        .map_err(|e| ToolError::Execution(anyhow::Error::new(e)))?;
        Ok(CandidatePrompt { record })
    }
    pub async fn grade(
        &self,
        candidate: &CandidatePrompt,
        grade: FeedbackGrade,
    ) -> Result<(), AgentError> {
        self.tools
            .call(
                "record_feedback",
                serde_json::json!({
                    "id": candidate.record.id,
                    "text": candidate.record.text,
                    "grade": grade,
                }),
            )
            .await?;
        Ok(())
    }

    async fn top_examples(
        &self,
        grade: FeedbackGrade,
        limit: usize,
    ) -> Result<Vec<crate::domain::FeedbackExample>, AgentError> {
        serde_json::from_value(
            self.tools
                .call(
                    "top_feedback_examples",
                    serde_json::json!({"grade": grade, "limit": limit}),
                )
                .await?,
        )
        .map_err(|e| ToolError::Execution(anyhow::Error::new(e)).into())
    }

    async fn discover(
        &self,
        limit: usize,
    ) -> Result<Vec<crate::domain::PromptDiscovery>, AgentError> {
        let discoveries: Vec<crate::domain::PromptDiscovery> = serde_json::from_value(
            self.tools
                .call(
                    "discover_prompts",
                    serde_json::json!({"query": self.target, "limit": limit}),
                )
                .await?,
        )
        .map_err(|e| ToolError::Execution(anyhow::Error::new(e)))?;
        Ok(discoveries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        domain::{stable_prompt_id, FeedbackExample, PromptDiscovery, PromptRecord},
        ports::*,
        prompts::RenderedPrompt,
    };
    use std::sync::Mutex;

    struct JsonTool {
        tool_name: &'static str,
        calls: Mutex<Vec<serde_json::Value>>,
        response: serde_json::Value,
    }
    #[async_trait::async_trait]
    impl Tool for JsonTool {
        fn name(&self) -> &'static str {
            self.tool_name
        }
        async fn execute(&self, input: serde_json::Value) -> Result<serde_json::Value, ToolError> {
            self.calls.lock().unwrap().push(input);
            Ok(self.response.clone())
        }
    }
    struct Llm;
    #[async_trait::async_trait]
    impl LlmProvider for Llm {
        async fn generate(
            &self,
            _: &RenderedPrompt,
            _: &PromptContext,
        ) -> Result<String, LlmError> {
            Ok("candidate".into())
        }
    }

    #[tokio::test]
    async fn iteration_uses_fixed_json_tools_and_persists_candidate() {
        let dir = tempfile::tempdir().unwrap();
        let system = dir.path().join("s.md");
        let template = dir.path().join("t.md");
        std::fs::write(&system, "system").unwrap();
        std::fs::write(&template, "{{target}}").unwrap();
        let assets = PromptAssets::load(&crate::config::PromptAssetConfig {
            system_path: system.display().to_string(),
            template_path: template.display().to_string(),
        })
        .unwrap();
        let positive = Arc::new(JsonTool {
            tool_name: "top_feedback_examples",
            calls: Mutex::new(vec![]),
            response: serde_json::to_value(Vec::<FeedbackExample>::new()).unwrap(),
        });
        let discover = Arc::new(JsonTool {
            tool_name: "discover_prompts",
            calls: Mutex::new(vec![]),
            response: serde_json::to_value(Vec::<PromptDiscovery>::new()).unwrap(),
        });
        let insert = Arc::new(JsonTool {
            tool_name: "insert_prompt",
            calls: Mutex::new(vec![]),
            response: serde_json::to_value(PromptRecord {
                id: stable_prompt_id("candidate"),
                text: "candidate".into(),
            })
            .unwrap(),
        });
        let record = Arc::new(JsonTool {
            tool_name: "record_feedback",
            calls: Mutex::new(vec![]),
            response: serde_json::json!({}),
        });
        let tools = Arc::new(
            ToolRegistry::new(vec![
                positive.clone(),
                discover.clone(),
                insert.clone(),
                record.clone(),
            ])
            .unwrap(),
        );
        let agent = Agent::new("target".into(), tools, Arc::new(Llm), assets);
        let result = agent
            .iterate(&crate::config::RetrievalConfig::default())
            .await
            .unwrap();
        assert_eq!(result.record.text, "candidate");
        assert_eq!(discover.calls.lock().unwrap()[0]["limit"], 2);
        assert_eq!(insert.calls.lock().unwrap()[0]["text"], "candidate");
        agent.grade(&result, FeedbackGrade::Positive).await.unwrap();
        assert_eq!(record.calls.lock().unwrap()[0]["text"], "candidate");
    }
}
