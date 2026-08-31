use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub type PromptId = Uuid;

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
    pub name: String,
    pub description: String,
    pub knowledge: Knowledge,
}

impl SkillRecord {
    pub fn positive_vector_source(&self) -> String {
        vector_source(&self.usage.when_to_use, &self.usage.signals)
    }

    pub fn negative_vector_source(&self) -> String {
        vector_source(&self.usage.when_not_to_use, &self.usage.anti_signals)
    }

    pub fn model_projection(&self) -> SkillProjection {
        SkillProjection {
            name: self.name.clone(),
            description: self.description.clone(),
            knowledge: self.knowledge.clone(),
        }
    }
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
        assert_eq!(value.as_object().unwrap().len(), 3);
        assert!(value.get("name").is_some());
        assert!(value.get("description").is_some());
        assert!(value.get("knowledge").is_some());
        assert!(value.get("usage").is_none());
        assert!(value.get("lifecycle").is_none());
    }
}
