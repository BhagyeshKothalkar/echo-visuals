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
            let text_tokens = tokens(&record.text);
            let score = if query_tokens.is_empty() {
                0.0
            } else {
                query_tokens.intersection(&text_tokens).count() as f32 / query_tokens.len() as f32
            };
            SkillDiscovery { record, score }
        })
        .collect();
    result.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.record.id.cmp(&b.record.id))
    });
    result.truncate(limit);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{stable_prompt_id, SkillRecord};
    #[test]
    fn ranks_by_case_insensitive_token_overlap_and_limit() {
        let records = ["Rust async agent", "Cooking recipe", "Rust testing"]
            .into_iter()
            .map(|text| SkillRecord {
                id: stable_prompt_id(text),
                text: text.into(),
            });
        let result = rank("rust agent", records, 2);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].record.text, "Rust async agent");
    }
}
