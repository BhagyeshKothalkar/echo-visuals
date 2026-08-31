use crate::config::PromptAssetConfig;
use std::fs;

#[derive(Debug, thiserror::Error)]
pub enum PromptError {
    #[error("could not read prompt asset: {0}")]
    Read(#[source] std::io::Error),
    #[error("prompt asset is empty: {0}")]
    Empty(String),
}
#[derive(Clone, Debug)]
pub struct RenderedPrompt {
    pub system: String,
    pub user: String,
}
#[derive(Clone, Debug)]
pub struct PromptAssets {
    system: String,
    template: String,
}
impl PromptAssets {
    pub fn load(config: &PromptAssetConfig) -> Result<Self, PromptError> {
        let system = fs::read_to_string(&config.system_path).map_err(PromptError::Read)?;
        let template = fs::read_to_string(&config.template_path).map_err(PromptError::Read)?;
        if system.trim().is_empty() {
            return Err(PromptError::Empty(config.system_path.clone()));
        }
        if template.trim().is_empty() {
            return Err(PromptError::Empty(config.template_path.clone()));
        }
        Ok(Self { system, template })
    }
    pub fn render(&self, target: &str) -> RenderedPrompt {
        let user = self.template.replace("{{target}}", target);
        RenderedPrompt {
            system: self.system.clone(),
            user,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::PromptAssetConfig;
    use std::io::Write;

    #[test]
    fn loads_and_renders_editable_markdown_assets() {
        let dir = tempfile::tempdir().unwrap();
        let system = dir.path().join("system.md");
        let template = dir.path().join("template.md");
        std::fs::File::create(&system)
            .unwrap()
            .write_all(b"system")
            .unwrap();
        std::fs::File::create(&template)
            .unwrap()
            .write_all(b"Target: {{target}}")
            .unwrap();
        let assets = PromptAssets::load(&PromptAssetConfig {
            system_path: system.display().to_string(),
            template_path: template.display().to_string(),
        })
        .unwrap();
        let rendered = assets.render("target text");
        assert_eq!(rendered.user, "Target: target text");
    }
}
