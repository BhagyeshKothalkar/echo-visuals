use serde::Deserialize;
use std::{fs, path::Path};

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("could not read config: {0}")]
    Read(#[source] std::io::Error),
    #[error("invalid config: {0}")]
    Parse(#[source] toml::de::Error),
    #[error("invalid setting: {0}")]
    Invalid(String),
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct Config {
    pub qdrant: QdrantConfig,
    pub redis: RedisConfig,
    pub llm: LlmConfig,
    pub prompts: PromptAssetConfig,
    pub logging: LoggingConfig,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct QdrantConfig {
    pub url: String,
    pub collection: String,
    pub seed: bool,
    pub embedding_dimension: usize,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct RedisConfig {
    pub url: String,
    pub key_prefix: String,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct LlmConfig {
    pub base_url: String,
    pub model: String,
    pub temperature: f32,
    pub max_tokens: u32,
    pub api_key: Option<String>,
    pub organization: Option<String>,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct PromptAssetConfig {
    pub system_path: String,
    pub template_path: String,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct LoggingConfig {
    pub level: String,
}

impl Default for QdrantConfig {
    fn default() -> Self {
        Self {
            url: "http://localhost:6334".into(),
            collection: "prompt_improver".into(),
            seed: true,
            embedding_dimension: 1,
        }
    }
}
impl Default for RedisConfig {
    fn default() -> Self {
        Self {
            url: "redis://127.0.0.1/".into(),
            key_prefix: "prompt-improver".into(),
        }
    }
}
impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            base_url: "https://api.openai.com/v1".into(),
            model: "gpt-4o-mini".into(),
            temperature: 0.2,
            max_tokens: 512,
            api_key: None,
            organization: None,
        }
    }
}
impl Default for PromptAssetConfig {
    fn default() -> Self {
        Self {
            system_path: "prompts/agents/prompt-improver-system.md".into(),
            template_path: "prompts/templates/candidate-request.md".into(),
        }
    }
}
impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".into(),
        }
    }
}

impl Config {
    pub fn from_sources(path: Option<&Path>) -> Result<Self, ConfigError> {
        dotenvy::dotenv().ok();
        let mut config = Config::default();
        if let Some(path) = path {
            let raw = fs::read_to_string(path).map_err(ConfigError::Read)?;
            let file: Config = toml::from_str(&raw).map_err(ConfigError::Parse)?;
            config = file;
        }
        config.llm.api_key = std::env::var("LLM_API_KEY").ok().or(config.llm.api_key);
        config.llm.organization = std::env::var("LLM_ORGANIZATION")
            .ok()
            .or(config.llm.organization);
        config.validate()?;
        Ok(config)
    }
    fn validate(&self) -> Result<(), ConfigError> {
        if !(0.0..=2.0).contains(&self.llm.temperature) {
            return Err(ConfigError::Invalid(
                "temperature must be between 0 and 2".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    #[test]
    fn toml_overrides_defaults() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        writeln!(f, "[llm]\nmodel = 'local-model'").unwrap();
        let c = Config::from_sources(Some(f.path())).unwrap();
        assert_eq!(c.llm.model, "local-model");
    }
}
