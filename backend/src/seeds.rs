use crate::domain::{stable_prompt_id, Knowledge, Lifecycle, Retrieval, SkillRecord, Usage};

pub fn seed_skills() -> Vec<SkillRecord> {
    [
        (
            "Concise explanations",
            "Write a concise technical explanation with one practical example.",
        ),
        (
            "Rigorous editing",
            "Act as a rigorous editor: improve clarity, structure, and factual precision.",
        ),
        (
            "Explicit assumptions",
            "Return a helpful answer with assumptions stated explicitly and no invented facts.",
        ),
        (
            "Actionable analysis",
            "Analyze the request, identify constraints, and propose an actionable solution.",
        ),
    ]
    .into_iter()
    .map(|(name, core)| SkillRecord {
        skill_id: stable_prompt_id(name),
        name: name.into(),
        description: core.into(),
        knowledge: Knowledge {
            core: core.into(),
            principles: vec![],
            procedures: vec![],
            failure_modes: vec![],
            examples: vec![],
        },
        usage: Usage {
            when_to_use: "When improving a response".into(),
            when_not_to_use: String::new(),
            signals: vec!["improve".into()],
            anti_signals: vec![],
        },
        retrieval: Retrieval {
            keywords: vec!["prompt".into(), "response".into()],
        },
        lifecycle: Lifecycle {
            version: "2.0".into(),
            status: "active".into(),
            source: "seed".into(),
            confidence: 1.0,
        },
    })
    .collect()
}
