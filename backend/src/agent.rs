use crate::{
    domain::{CandidatePrompt, FeedbackGrade},
    ports::{GenerationResult, LlmError, LlmProvider, ToolError, ToolRegistry},
    prompts::PromptAssets,
};
use std::sync::Arc;
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error(transparent)]
    Tool(#[from] ToolError),
    #[error(transparent)]
    Llm(#[from] LlmError),
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
    pub async fn iterate(&self) -> Result<GenerationResult, AgentError> {
        let rendered = self.assets.render(&self.target);
        Ok(self.llm.generate(&rendered, self.tools.clone()).await?)
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
                    "id": candidate.id,
                    "text": candidate.text,
                    "grade": grade,
                }),
            )
            .await?;
        Ok(())
    }

    pub async fn save_skill(&self, text: &str) -> Result<crate::domain::SkillRecord, AgentError> {
        serde_json::from_value(
            self.tools
                .call("save_skill", serde_json::json!({"text": text}))
                .await?,
        )
        .map_err(|e| crate::ports::ToolError::Execution(anyhow::Error::new(e)).into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        domain::{stable_prompt_id, SkillRecord},
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
            _: Arc<ToolRegistry>,
        ) -> Result<GenerationResult, LlmError> {
            Ok(GenerationResult {
                text: "candidate".into(),
                ..Default::default()
            })
        }
    }

    #[tokio::test]
    async fn iteration_uses_model_output_as_candidate_for_feedback() {
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
            tool_name: "get_feedback",
            calls: Mutex::new(vec![]),
            response: serde_json::json!([]),
        });
        let discover = Arc::new(JsonTool {
            tool_name: "search_skills",
            calls: Mutex::new(vec![]),
            response: serde_json::json!([]),
        });
        let insert = Arc::new(JsonTool {
            tool_name: "save_skill",
            calls: Mutex::new(vec![]),
            response: serde_json::to_value(SkillRecord {
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
        let result = agent.iterate().await.unwrap();
        assert_eq!(result.text, "candidate");
        assert_eq!(result.stats.total_calls, 0);
        let candidate = CandidatePrompt {
            id: stable_prompt_id(&result.text),
            text: result.text,
        };
        agent
            .grade(&candidate, FeedbackGrade::Positive)
            .await
            .unwrap();
        assert_eq!(record.calls.lock().unwrap()[0]["text"], "candidate");
    }
}
