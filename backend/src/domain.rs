use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub type PromptId = Uuid;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalystInput {
    pub target: String,
    pub image: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SkillRecord {
    pub skill_id: PromptId,
    pub name: String,
    pub description: String,
    pub knowledge: Knowledge,
    pub usage: Usage,
    pub retrieval: Retrieval,
    pub lifecycle: Lifecycle,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Knowledge {
    pub core: String,
    pub principles: Vec<String>,
    pub procedures: Vec<String>,
    pub failure_modes: Vec<String>,
    pub examples: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub when_to_use: String,
    pub when_not_to_use: String,
    pub signals: Vec<String>,
    pub anti_signals: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Retrieval {
    pub keywords: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Lifecycle {
    pub version: String,
    pub status: String,
    pub source: String,
    pub confidence: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SkillProjection {
    pub skill_id: PromptId,
    pub name: String,
    pub description: String,
    pub knowledge: Knowledge,
}

impl SkillRecord {
    pub fn validate(&self) -> Result<(), String> {
        if self.skill_id.is_nil() {
            return Err("skill_id must not be nil".into());
        }
        if self.name.trim().is_empty() || self.description.trim().is_empty() {
            return Err("name and description must not be empty".into());
        }
        if self.knowledge.core.trim().is_empty() {
            return Err("knowledge.core must not be empty".into());
        }
        Ok(())
    }
    pub fn positive_vector_source(&self) -> String {
        vector_source(&self.usage.when_to_use, &self.usage.signals)
    }

    pub fn negative_vector_source(&self) -> String {
        vector_source(&self.usage.when_not_to_use, &self.usage.anti_signals)
    }

    pub fn model_projection(&self) -> SkillProjection {
        SkillProjection {
            skill_id: self.skill_id,
            name: self.name.clone(),
            description: self.description.clone(),
            knowledge: self.knowledge.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnalystOutput {
    pub observations: Vec<String>,
    pub weaknesses: Vec<String>,
    pub requirements: Vec<String>,
    pub relevant_skill_ids: Vec<PromptId>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OptimizerOutput {
    pub prompt: String,
    pub save_skill: bool,
    pub skill_reason: Option<String>,
}

fn vector_source(context: &str, signals: &[String]) -> String {
    std::iter::once(context)
        .chain(signals.iter().map(String::as_str))
        .filter(|part| !part.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

pub type SkillDiscovery = SkillProjection;

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
    pub id: PromptId,
    pub text: String,
}

pub fn stable_prompt_id(text: &str) -> PromptId {
    Uuid::new_v5(
        &Uuid::NAMESPACE_URL,
        format!("prompt:{}", text.trim()).as_bytes(),
    )
}

#[cfg(test)]
mod v2_tests {
    use super::*;

    fn record() -> SkillRecord {
        SkillRecord {
            skill_id: Uuid::nil(),
            name: "Structured answers".into(),
            description: "Improve clarity".into(),
            knowledge: Knowledge {
                core: "Use a clear structure".into(),
                principles: vec!["Be precise".into()],
                procedures: vec!["State assumptions".into()],
                failure_modes: vec!["Inventing facts".into()],
                examples: vec!["A short example".into()],
            },
            usage: Usage {
                when_to_use: "When the request needs explanation".into(),
                when_not_to_use: "When the user wants raw output".into(),
                signals: vec!["explain".into()],
                anti_signals: vec!["just output".into()],
            },
            retrieval: Retrieval {
                keywords: vec!["clarity".into()],
            },
            lifecycle: Lifecycle {
                version: "2.0".into(),
                status: "active".into(),
                source: "seed".into(),
                confidence: 0.9,
            },
        }
    }

    #[test]
    fn v2_record_serializes_with_canonical_fields() {
        let value = serde_json::to_value(record()).unwrap();
        assert!(value.get("skill_id").is_some());
        assert!(value.get("knowledge").is_some());
        assert!(value.get("usage").is_some());
        assert!(value.get("retrieval").is_some());
        assert!(value.get("lifecycle").is_some());
        assert!(value.get("text").is_none());
    }

    #[test]
    fn vector_sources_use_only_their_matching_usage_fields() {
        let skill = record();
        assert_eq!(
            skill.positive_vector_source(),
            "When the request needs explanation\nexplain"
        );
        assert_eq!(
            skill.negative_vector_source(),
            "When the user wants raw output\njust output"
        );
    }

    #[test]
    fn model_projection_exposes_only_name_description_and_knowledge() {
        let value = serde_json::to_value(record().model_projection()).unwrap();
        assert_eq!(value.as_object().unwrap().len(), 4);
        assert!(value.get("name").is_some());
        assert!(value.get("description").is_some());
        assert!(value.get("knowledge").is_some());
        assert!(value.get("usage").is_none());
        assert!(value.get("lifecycle").is_none());
    }

    #[test]
    fn analyst_and_optimizer_contracts_reject_missing_required_fields() {
        assert!(serde_json::from_value::<AnalystOutput>(serde_json::json!({})).is_err());
        assert!(
            serde_json::from_value::<OptimizerOutput>(serde_json::json!({"prompt":"x"})).is_err()
        );
    }

    #[test]
    fn curated_skills_require_canonical_nonempty_identity_and_knowledge() {
        let mut skill = record();
        skill.skill_id = Uuid::nil();
        assert!(skill.validate().is_err());
        skill.skill_id = Uuid::new_v5(&Uuid::NAMESPACE_URL, b"valid");
        assert!(skill.validate().is_ok());
    }
}
