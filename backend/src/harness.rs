use crate::{agent::Agent, config::RetrievalConfig, domain::CandidatePrompt};
#[derive(Debug, thiserror::Error)]
pub enum HarnessError {
    #[error(transparent)]
    Agent(#[from] crate::agent::AgentError),
    #[error("model returned a candidate without saving a skill")]
    MissingSavedSkill,
}
#[derive(Clone, Debug)]
pub struct Harness {
    pub retrieval: RetrievalConfig,
}
impl Harness {
    pub fn new(retrieval: RetrievalConfig) -> Self {
        Self { retrieval }
    }
    pub async fn run_iteration(&self, agent: &Agent) -> Result<CandidatePrompt, HarnessError> {
        let result = agent.iterate(&self.retrieval).await?;
        let record = match result.saved_skill {
            Some(record) => record,
            None if result.stats.search_skill_calls > 2 => agent.save_skill(&result.text).await?,
            None => return Err(HarnessError::MissingSavedSkill),
        };
        Ok(CandidatePrompt { record })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        domain::{stable_prompt_id, PromptRecord},
        ports::{
            GenerationResult, LlmError, LlmProvider, Tool, ToolCallStats, ToolError, ToolRegistry,
        },
        prompts::{PromptAssets, PromptContext, RenderedPrompt},
    };
    use async_trait::async_trait;
    use std::{
        collections::HashMap,
        sync::{Arc, Mutex},
    };

    struct ToolStub {
        name: &'static str,
        calls: Arc<Mutex<usize>>,
        response: serde_json::Value,
    }
    #[async_trait]
    impl Tool for ToolStub {
        fn name(&self) -> &'static str {
            self.name
        }
        async fn execute(&self, _: serde_json::Value) -> Result<serde_json::Value, ToolError> {
            *self.calls.lock().unwrap() += 1;
            Ok(self.response.clone())
        }
    }
    struct Model;
    #[async_trait]
    impl LlmProvider for Model {
        async fn generate(
            &self,
            _: &RenderedPrompt,
            _: &PromptContext,
            _: Arc<ToolRegistry>,
        ) -> Result<GenerationResult, LlmError> {
            Ok(GenerationResult {
                text: "fallback skill".into(),
                stats: ToolCallStats {
                    total_calls: 3,
                    search_skill_calls: 3,
                    saved_skill: false,
                },
                saved_skill: None,
            })
        }
    }

    #[tokio::test]
    async fn three_searches_without_explicit_save_trigger_one_fallback_save() {
        let dir = tempfile::tempdir().unwrap();
        let system = dir.path().join("system.md");
        let template = dir.path().join("template.md");
        std::fs::write(&system, "system").unwrap();
        std::fs::write(&template, "{{target}}").unwrap();
        let assets = PromptAssets::load(&crate::config::PromptAssetConfig {
            system_path: system.display().to_string(),
            template_path: template.display().to_string(),
        })
        .unwrap();
        let saved = Arc::new(Mutex::new(0));
        let save = ToolStub {
            name: "save_skill",
            calls: saved.clone(),
            response: serde_json::to_value(PromptRecord {
                id: stable_prompt_id("fallback skill"),
                text: "fallback skill".into(),
            })
            .unwrap(),
        };
        let mut tools = HashMap::new();
        tools.insert("save_skill", Arc::new(save) as Arc<dyn Tool>);
        let registry = Arc::new(ToolRegistry::new(tools.into_values().collect()).unwrap());
        let agent = Agent::new("target".into(), registry, Arc::new(Model), assets);
        let candidate = Harness::new(RetrievalConfig::default())
            .run_iteration(&agent)
            .await
            .unwrap();
        assert_eq!(candidate.record.text, "fallback skill");
        assert_eq!(*saved.lock().unwrap(), 1);
    }
}
