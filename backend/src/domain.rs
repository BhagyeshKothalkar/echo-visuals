use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub type PromptId = Uuid;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptRecord {
    pub id: PromptId,
    pub text: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PromptDiscovery {
    pub record: PromptRecord,
    pub score: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeedbackGrade {
    Positive,
    Negative,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeedbackExample {
    pub id: PromptId,
    pub text: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CandidatePrompt {
    pub record: PromptRecord,
}

pub fn stable_prompt_id(text: &str) -> PromptId {
    Uuid::new_v5(
        &Uuid::NAMESPACE_URL,
        format!("prompt:{}", text.trim()).as_bytes(),
    )
}
