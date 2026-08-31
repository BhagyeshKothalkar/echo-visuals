use crate::{agent::Agent, domain::CandidatePrompt};
#[derive(Debug, thiserror::Error)]
pub enum HarnessError {
    #[error(transparent)]
    Agent(#[from] crate::agent::AgentError),
}
#[derive(Clone, Debug, Default)]
pub struct Harness;
impl Harness {
    pub fn new() -> Self {
        Self
    }

    pub async fn run_iteration(&self, agent: &Agent) -> Result<CandidatePrompt, HarnessError> {
        let result = agent.iterate().await?;
        Ok(CandidatePrompt {
            id: crate::domain::stable_prompt_id(&result.text),
            text: result.text,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        domain::{stable_prompt_id, Knowledge, Lifecycle, Retrieval, SkillRecord, Usage},
        ports::{
            GenerationResult, LlmError, LlmProvider, Tool, ToolCallStats, ToolError, ToolRegistry,
        },
        prompts::{PromptAssets, RenderedPrompt},
    };
    use async_trait::async_trait;
    use std::sync::{Arc, Mutex};

    fn skill(name: &str) -> SkillRecord {
        SkillRecord {
            skill_id: stable_prompt_id(name),
            name: name.into(),
            description: name.into(),
            knowledge: Knowledge {
                core: name.into(),
                principles: vec![],
                procedures: vec![],
                failure_modes: vec![],
                examples: vec![],
            },
            usage: Usage {
                when_to_use: name.into(),
                when_not_to_use: String::new(),
                signals: vec![],
                anti_signals: vec![],
            },
            retrieval: Retrieval { keywords: vec![] },
            lifecycle: Lifecycle {
                version: "2".into(),
                status: "active".into(),
                source: "test".into(),
                confidence: 1.0,
            },
        }
    }

    struct ToolStub {
        name: &'static str,
        calls: Arc<Mutex<usize>>,
        last_input: Option<Arc<Mutex<Option<serde_json::Value>>>>,
        response: serde_json::Value,
    }
    #[async_trait]
    impl Tool for ToolStub {
        fn name(&self) -> &'static str {
            self.name
        }
        async fn execute(&self, input: serde_json::Value) -> Result<serde_json::Value, ToolError> {
            *self.calls.lock().unwrap() += 1;
            if let Some(last_input) = &self.last_input {
                *last_input.lock().unwrap() = Some(input);
            }
            Ok(self.response.clone())
        }
    }
    struct Model {
        searches: usize,
        saved_skill: Option<SkillRecord>,
    }
    #[async_trait]
    impl LlmProvider for Model {
        async fn generate(
            &self,
            _: &RenderedPrompt,
            _: Arc<ToolRegistry>,
        ) -> Result<GenerationResult, LlmError> {
            Ok(GenerationResult {
                text: "candidate output".into(),
                stats: ToolCallStats {
                    total_calls: self.searches,
                    search_skill_calls: self.searches,
                    saved_skill: self.saved_skill.is_some(),
                },
                saved_skill: self.saved_skill.clone(),
            })
        }
    }

    fn assets(dir: &tempfile::TempDir) -> PromptAssets {
        let system = dir.path().join("system.md");
        let template = dir.path().join("template.md");
        std::fs::write(&system, "system").unwrap();
        std::fs::write(&template, "{{target}}").unwrap();
        PromptAssets::load(&crate::config::PromptAssetConfig {
            system_path: system.display().to_string(),
            template_path: template.display().to_string(),
        })
        .unwrap()
    }

    #[tokio::test]
    async fn candidate_is_model_output_without_saved_skill() {
        let dir = tempfile::tempdir().unwrap();
        let assets = assets(&dir);
        let registry = Arc::new(ToolRegistry::new(vec![]).unwrap());
        let agent = Agent::new(
            "target".into(),
            registry,
            Arc::new(Model {
                searches: 1,
                saved_skill: None,
            }),
            assets,
        );

        let candidate = Harness::new().run_iteration(&agent).await.unwrap();

        assert_eq!(candidate.text, "candidate output");
        assert_eq!(candidate.id, stable_prompt_id("candidate output"));
    }

    #[tokio::test]
    async fn explicit_saved_skill_does_not_replace_model_output_candidate() {
        let dir = tempfile::tempdir().unwrap();
        let assets = assets(&dir);
        let saved = SkillRecord {
            ..skill("explicit skill")
        };
        let save_calls = Arc::new(Mutex::new(0));
        let save = ToolStub {
            name: "save_skill",
            calls: save_calls.clone(),
            last_input: None,
            response: serde_json::to_value(saved.clone()).unwrap(),
        };
        let registry = Arc::new(ToolRegistry::new(vec![Arc::new(save) as Arc<dyn Tool>]).unwrap());
        let agent = Agent::new(
            "target".into(),
            registry,
            Arc::new(Model {
                searches: 3,
                saved_skill: Some(saved.clone()),
            }),
            assets,
        );

        let candidate = Harness::new().run_iteration(&agent).await.unwrap();

        assert_eq!(*save_calls.lock().unwrap(), 0);
        assert_eq!(candidate.id, stable_prompt_id("candidate output"));
        assert_eq!(candidate.text, "candidate output");
    }
}
