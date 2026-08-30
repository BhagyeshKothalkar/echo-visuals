use crate::{
    config::RetrievalConfig,
    domain::{CandidatePrompt, FeedbackGrade},
    ports::{FeedbackStore, LlmError, LlmProvider, PromptRepository, StorageError},
    prompts::{PromptAssets, PromptContext},
};
use std::sync::Arc;
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error(transparent)]
    Storage(#[from] StorageError),
    #[error(transparent)]
    Llm(#[from] LlmError),
    #[error("prompt rendering failed: {0}")]
    Prompt(String),
}
pub struct Agent {
    target: String,
    repo: Arc<dyn PromptRepository>,
    feedback: Arc<dyn FeedbackStore>,
    llm: Arc<dyn LlmProvider>,
    assets: PromptAssets,
}
impl Agent {
    pub fn new(
        target: String,
        repo: Arc<dyn PromptRepository>,
        feedback: Arc<dyn FeedbackStore>,
        llm: Arc<dyn LlmProvider>,
        assets: PromptAssets,
    ) -> Self {
        Self {
            target,
            repo,
            feedback,
            llm,
            assets,
        }
    }
    pub async fn iterate(&self, limits: &RetrievalConfig) -> Result<CandidatePrompt, AgentError> {
        let positive = self
            .feedback
            .top_examples(FeedbackGrade::Positive, limits.positive_limit)
            .await?;
        let negative = self
            .feedback
            .top_examples(FeedbackGrade::Negative, limits.negative_limit)
            .await?;
        let discoveries = self
            .repo
            .discover(&self.target, limits.discovery_limit)
            .await?;
        let context = PromptContext {
            target: self.target.clone(),
            positive,
            negative,
            discoveries,
        };
        let rendered = self.assets.render(&context);
        let text = self.llm.generate(&rendered, &context).await?;
        let record = self.repo.insert_prompt(&text).await?;
        Ok(CandidatePrompt { record })
    }
    pub async fn grade(
        &self,
        candidate: &CandidatePrompt,
        grade: FeedbackGrade,
    ) -> Result<(), AgentError> {
        self.feedback
            .record_feedback(candidate, grade)
            .await
            .map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        domain::{stable_prompt_id, PromptDiscovery, PromptRecord},
        ports::*,
        prompts::RenderedPrompt,
    };
    use std::sync::Mutex;

    struct Repo {
        inserted: Mutex<Vec<String>>,
    }
    #[async_trait::async_trait]
    impl PromptRepository for Repo {
        async fn initialize(&self) -> Result<(), StorageError> {
            Ok(())
        }
        async fn insert_prompt(&self, text: &str) -> Result<PromptRecord, StorageError> {
            self.inserted.lock().unwrap().push(text.into());
            Ok(PromptRecord {
                id: stable_prompt_id(text),
                text: text.into(),
            })
        }
        async fn discover(
            &self,
            _: &str,
            limit: usize,
        ) -> Result<Vec<PromptDiscovery>, StorageError> {
            assert_eq!(limit, 2);
            Ok(vec![])
        }
    }
    struct Feedback;
    #[async_trait::async_trait]
    impl FeedbackStore for Feedback {
        async fn record_feedback(
            &self,
            _: &CandidatePrompt,
            _: FeedbackGrade,
        ) -> Result<(), StorageError> {
            Ok(())
        }
        async fn top_examples(
            &self,
            _: FeedbackGrade,
            limit: usize,
        ) -> Result<Vec<crate::domain::FeedbackExample>, StorageError> {
            assert_eq!(limit, 2);
            Ok(vec![])
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
    async fn iteration_retrieves_two_examples_and_persists_candidate() {
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
        let repo = Arc::new(Repo {
            inserted: Mutex::new(vec![]),
        });
        let agent = Agent::new(
            "target".into(),
            repo.clone(),
            Arc::new(Feedback),
            Arc::new(Llm),
            assets,
        );
        let result = agent
            .iterate(&crate::config::RetrievalConfig::default())
            .await
            .unwrap();
        assert_eq!(result.record.text, "candidate");
        assert_eq!(repo.inserted.lock().unwrap().as_slice(), &["candidate"]);
    }
}
