use crate::domain::{SkillDiscovery, SkillRecord};
use std::collections::HashSet;

pub fn tokens(text: &str) -> HashSet<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_owned)
        .collect()
}

pub fn rank(
    query: &str,
    records: impl IntoIterator<Item = SkillRecord>,
    limit: usize,
) -> Vec<SkillDiscovery> {
    let query_tokens = tokens(query);
    let mut result: Vec<_> = records
        .into_iter()
        .map(|record| {
            let searchable = format!(
                "{} {} {} {} {:?}",
                record.name,
                record.description,
                record.knowledge.core,
                record.usage.when_to_use,
                record.retrieval.keywords
            );
            let text_tokens = tokens(&searchable);
            let score = if query_tokens.is_empty() {
                0.0
            } else {
                query_tokens.intersection(&text_tokens).count() as f32 / query_tokens.len() as f32
            };
            (record.model_projection(), score, record.skill_id)
        })
        .collect();
    result.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.2.cmp(&b.2))
    });
    result.truncate(limit);
    result.into_iter().map(|(skill, _, _)| skill).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{Knowledge, Lifecycle, Retrieval, Usage};
    #[test]
    fn ranks_by_case_insensitive_token_overlap_and_limit() {
        let records = ["Rust async agent", "Cooking recipe", "Rust testing"]
            .into_iter()
            .map(|name| crate::domain::SkillRecord {
                skill_id: crate::domain::stable_prompt_id(name),
                name: name.into(),
                description: String::new(),
                knowledge: Knowledge {
                    core: String::new(),
                    principles: vec![],
                    procedures: vec![],
                    failure_modes: vec![],
                    examples: vec![],
                },
                usage: Usage {
                    when_to_use: String::new(),
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
            });
        let result = rank("rust agent", records, 2);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].name, "Rust async agent");
    }
}
