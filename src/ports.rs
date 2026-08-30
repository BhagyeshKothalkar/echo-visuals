use crate::{
    domain::{CandidatePrompt, FeedbackExample, FeedbackGrade, PromptDiscovery, PromptRecord},
    prompts::{PromptContext, RenderedPrompt},
};
use async_trait::async_trait;

#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    #[error("storage backend error: {0}")]
    Backend(#[source] anyhow::Error),
    #[error("invalid stored data: {0}")]
    Invalid(String),
}

#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    #[error("LLM request failed: {0}")]
    Request(#[source] anyhow::Error),
    #[error("LLM returned an empty candidate")]
    EmptyCandidate,
}

#[async_trait]
pub trait PromptRepository: Send + Sync {
    async fn initialize(&self) -> Result<(), StorageError>;
    async fn insert_prompt(&self, text: &str) -> Result<PromptRecord, StorageError>;
    async fn discover(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<PromptDiscovery>, StorageError>;
}

#[async_trait]
pub trait FeedbackStore: Send + Sync {
    async fn record_feedback(
        &self,
        candidate: &CandidatePrompt,
        grade: FeedbackGrade,
    ) -> Result<(), StorageError>;
    async fn top_examples(
        &self,
        grade: FeedbackGrade,
        limit: usize,
    ) -> Result<Vec<FeedbackExample>, StorageError>;
}

#[async_trait]
pub trait LlmProvider: Send + Sync {
    async fn generate(
        &self,
        prompt: &RenderedPrompt,
        context: &PromptContext,
    ) -> Result<String, LlmError>;
}
