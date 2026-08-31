use crate::domain::{stable_prompt_id, Knowledge, Lifecycle, Retrieval, SkillRecord, Usage};

pub fn seed_skills() -> Vec<SkillRecord> {
    [
        (
            "Subject Composition",
            "Compose the requested image subject with clear framing, hierarchy, perspective, and spatial balance.",
        ),
        (
            "Lighting Control",
            "Specify and control lighting direction, quality, intensity, color temperature, and resulting shadows.",
        ),
        (
            "Style Direction",
            "Translate the requested visual style into concrete choices for medium, rendering approach, texture, palette, and aesthetic.",
        ),
        (
            "Photorealistic Rendering",
            "Improve image prompts for realistic materials, natural lighting, believable proportions, and photographic detail.",
        ),
        (
            "Camera Control",
            "Specify camera position, lens characteristics, focal length, depth of field, focus, and framing.",
        ),
        (
            "Image Inpainting",
            "Edit a localized region of an image while preserving surrounding context, geometry, lighting, and visual consistency.",
        ),
        (
            "Object Removal",
            "Remove an unwanted object from an image and reconstruct the affected region so it blends naturally with the surrounding scene.",
        ),
        (
            "Object Replacement",
            "Replace a specific object while preserving the original scene composition, lighting, perspective, and environmental context.",
        ),
        (
            "Background Replacement",
            "Replace or redesign the image background while maintaining subject identity, edges, lighting, and scene integration.",
        ),
        (
            "Pose and Structure Editing",
            "Modify the pose, anatomy, orientation, or spatial arrangement of a subject while preserving identity and overall visual coherence.",
        ),
        (
            "Image Expansion",
            "Extend an image beyond its original boundaries while generating contextually consistent content that matches composition, perspective, and style.",
        ),
        (
            "Detail Enhancement",
            "Increase useful visual detail and local fidelity without introducing artifacts, unwanted objects, or inconsistent textures.",
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
            when_to_use: "When generating or editing images".into(),
            when_not_to_use: String::new(),
            signals: vec!["image".into(), "generate".into(), "edit".into()],
            anti_signals: vec![],
        },
        retrieval: Retrieval {
            keywords: vec![
                "image".into(),
                "generation".into(),
                "editing".into(),
            ],
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
