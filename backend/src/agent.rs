use crate::{
    domain::{AnalystInput, CandidatePrompt, FeedbackGrade, SkillRecord},
    ports::{GenerationResult, LlmProvider, SkillStore, StorageError, ToolError, ToolRegistry},
};
use std::{collections::HashSet, sync::Arc};
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error(transparent)]
    Tool(#[from] ToolError),
    #[error(transparent)]
    Llm(#[from] crate::ports::LlmError),
    #[error(transparent)]
    Storage(#[from] StorageError),
    #[error("analyst selected missing skill {0}")]
    MissingSkill(uuid::Uuid),
}
pub struct Agent {
    target: String,
    image: String,
    tools: Arc<ToolRegistry>,
    skills: Arc<dyn SkillStore>,
    llm: Arc<dyn LlmProvider>,
}
impl Agent {
    pub fn new(
        target: String,
        image: String,
        tools: Arc<ToolRegistry>,
        skills: Arc<dyn SkillStore>,
        llm: Arc<dyn LlmProvider>,
    ) -> Self {
        Self {
            target,
            image,
            tools,
            skills,
            llm,
        }
    }
    pub async fn iterate(&self) -> Result<GenerationResult, AgentError> {
        let analysis = self
            .llm
            .analyze(
                &AnalystInput {
                    target: self.target.clone(),
                    image: self.image.clone(),
                },
                self.tools.clone(),
            )
            .await?;
        let skills = self.resolve(&analysis.relevant_skill_ids).await?;
        let input =
            serde_json::json!({"target":self.target,"analysis":analysis,"selected_skills":skills})
                .to_string();
        let optimized = self.llm.optimize(&input).await?;
        if optimized.prompt.trim().is_empty() {
            return Err(crate::ports::LlmError::InvalidOutput {
                role: "optimizer".into(),
                message: "prompt must not be empty".into(),
            }
            .into());
        }
        let saved = if optimized.save_skill {
            let input=serde_json::json!({"target":self.target,"analysis":analysis,"prompt":optimized.prompt,"skill_reason":optimized.skill_reason,"selected_skills":skills}).to_string();
            let skill = self.llm.curate(&input).await?;
            skill
                .validate()
                .map_err(|message| crate::ports::LlmError::InvalidOutput {
                    role: "curator".into(),
                    message,
                })?;
            self.skills.save_skill(&skill).await?;
            Some(skill)
        } else {
            None
        };
        Ok(GenerationResult {
            text: optimized.prompt,
            saved_skill: saved,
        })
    }
    async fn resolve(&self, ids: &[uuid::Uuid]) -> Result<Vec<SkillRecord>, AgentError> {
        let mut seen = HashSet::new();
        let mut result = Vec::new();
        for id in ids {
            if seen.insert(*id) {
                let skill = self.skills.get_skill(*id).await.map_err(|e| match e {
                    StorageError::Invalid(_) => AgentError::MissingSkill(*id),
                    other => AgentError::Storage(other),
                })?;
                result.push(skill)
            }
        }
        Ok(result)
    }
    pub async fn grade(
        &self,
        candidate: &CandidatePrompt,
        grade: FeedbackGrade,
    ) -> Result<(), AgentError> {
        self.tools
            .call(
                "record_feedback",
                serde_json::json!({"id":candidate.id,"text":candidate.text,"grade":grade}),
            )
            .await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{AnalystOutput, Knowledge, Lifecycle, OptimizerOutput, Retrieval, Usage};
    use async_trait::async_trait;
    use std::sync::Mutex;

    fn skill(id: uuid::Uuid) -> SkillRecord {
        SkillRecord {
            skill_id: id,
            name: "reusable".into(),
            description: "d".into(),
            knowledge: Knowledge {
                core: "core".into(),
                principles: vec![],
                procedures: vec![],
                failure_modes: vec![],
                examples: vec![],
            },
            usage: Usage {
                when_to_use: "use".into(),
                when_not_to_use: "avoid".into(),
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
    struct Store {
        records: Vec<SkillRecord>,
        gets: Mutex<Vec<uuid::Uuid>>,
        saves: Mutex<usize>,
    }
    #[async_trait]
    impl SkillStore for Store {
        async fn get_skill(&self, id: uuid::Uuid) -> Result<SkillRecord, StorageError> {
            self.gets.lock().unwrap().push(id);
            self.records
                .iter()
                .find(|s| s.skill_id == id)
                .cloned()
                .ok_or_else(|| StorageError::Invalid("missing".into()))
        }
        async fn save_skill(&self, _: &SkillRecord) -> Result<(), StorageError> {
            *self.saves.lock().unwrap() += 1;
            Ok(())
        }
    }
    struct Model {
        ids: Vec<uuid::Uuid>,
        save: bool,
        calls: Mutex<Vec<&'static str>>,
    }
    #[async_trait]
    impl LlmProvider for Model {
        async fn analyze(
            &self,
            _: &crate::domain::AnalystInput,
            _: Arc<ToolRegistry>,
        ) -> Result<AnalystOutput, crate::ports::LlmError> {
            self.calls.lock().unwrap().push("analyze");
            Ok(AnalystOutput {
                observations: vec![],
                weaknesses: vec![],
                requirements: vec![],
                relevant_skill_ids: self.ids.clone(),
            })
        }
        async fn optimize(&self, _: &str) -> Result<OptimizerOutput, crate::ports::LlmError> {
            self.calls.lock().unwrap().push("optimize");
            Ok(OptimizerOutput {
                prompt: "optimized".into(),
                save_skill: self.save,
                skill_reason: Some("generalizable".into()),
            })
        }
        async fn curate(&self, _: &str) -> Result<SkillRecord, crate::ports::LlmError> {
            self.calls.lock().unwrap().push("curate");
            Ok(skill(uuid::Uuid::new_v5(
                &uuid::Uuid::NAMESPACE_URL,
                b"curated",
            )))
        }
    }
    fn make(model: Arc<Model>, store: Arc<Store>) -> Agent {
        Agent::new(
            "target".into(),
            "image".into(),
            Arc::new(ToolRegistry::new(vec![]).unwrap()),
            store,
            model,
        )
    }
    #[tokio::test]
    async fn deduplicates_and_keeps_candidate_as_optimizer_prompt() {
        let id = uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_URL, b"selected");
        let store = Arc::new(Store {
            records: vec![skill(id)],
            gets: Mutex::new(vec![]),
            saves: Mutex::new(0),
        });
        let model = Arc::new(Model {
            ids: vec![id, id],
            save: false,
            calls: Mutex::new(vec![]),
        });
        let result = make(model.clone(), store.clone()).iterate().await.unwrap();
        assert_eq!(result.text, "optimized");
        assert_eq!(*store.gets.lock().unwrap(), vec![id]);
        assert_eq!(*model.calls.lock().unwrap(), vec!["analyze", "optimize"]);
    }
    #[tokio::test]
    async fn curator_and_save_are_conditional() {
        let id = uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_URL, b"selected");
        let store = Arc::new(Store {
            records: vec![skill(id)],
            gets: Mutex::new(vec![]),
            saves: Mutex::new(0),
        });
        let model = Arc::new(Model {
            ids: vec![id],
            save: true,
            calls: Mutex::new(vec![]),
        });
        make(model.clone(), store.clone()).iterate().await.unwrap();
        assert_eq!(*store.saves.lock().unwrap(), 1);
        assert_eq!(
            *model.calls.lock().unwrap(),
            vec!["analyze", "optimize", "curate"]
        );
    }
    #[tokio::test]
    async fn missing_skill_stops_before_optimizer() {
        let store = Arc::new(Store {
            records: vec![],
            gets: Mutex::new(vec![]),
            saves: Mutex::new(0),
        });
        let model = Arc::new(Model {
            ids: vec![uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_URL, b"missing")],
            save: true,
            calls: Mutex::new(vec![]),
        });
        assert!(matches!(
            make(model.clone(), store).iterate().await,
            Err(AgentError::MissingSkill(_))
        ));
        assert_eq!(*model.calls.lock().unwrap(), vec!["analyze"]);
    }
}
