use crate::prompts::{PromptContext, RenderedPrompt};
use async_trait::async_trait;
use serde_json::Value;
use std::{collections::HashMap, sync::Arc};

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
pub trait Tool: Send + Sync {
    fn name(&self) -> &'static str;
    async fn execute(&self, input: Value) -> Result<Value, ToolError>;
}

#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    #[error("duplicate tool: {0}")]
    Duplicate(String),
    #[error("tool not found: {0}")]
    NotFound(String),
    #[error("invalid input for tool {tool}: {source}")]
    InvalidInput {
        tool: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("tool execution failed: {0}")]
    Execution(#[source] anyhow::Error),
}

pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new(tools: Vec<Arc<dyn Tool>>) -> Result<Self, ToolError> {
        let mut registered = HashMap::new();
        for tool in tools {
            let name = tool.name().to_string();
            if registered.insert(name.clone(), tool).is_some() {
                return Err(ToolError::Duplicate(name));
            }
        }
        Ok(Self { tools: registered })
    }

    pub async fn call(&self, name: &str, input: Value) -> Result<Value, ToolError> {
        self.tools
            .get(name)
            .ok_or_else(|| ToolError::NotFound(name.to_string()))?
            .execute(input)
            .await
    }
}

#[async_trait]
pub trait LlmProvider: Send + Sync {
    async fn generate(
        &self,
        prompt: &RenderedPrompt,
        context: &PromptContext,
    ) -> Result<String, LlmError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Echo(&'static str);

    #[async_trait]
    impl Tool for Echo {
        fn name(&self) -> &'static str {
            self.0
        }

        async fn execute(&self, input: Value) -> Result<Value, ToolError> {
            Ok(input)
        }
    }

    #[tokio::test]
    async fn registry_dispatches_heterogeneous_tools_by_fixed_name() {
        let registry =
            ToolRegistry::new(vec![Arc::new(Echo("first")), Arc::new(Echo("second"))]).unwrap();

        assert_eq!(
            registry
                .call("first", serde_json::json!({"x": 1}))
                .await
                .unwrap(),
            serde_json::json!({"x": 1})
        );
        assert_eq!(
            registry
                .call("second", serde_json::json!("value"))
                .await
                .unwrap(),
            serde_json::json!("value")
        );
    }

    #[test]
    fn registry_rejects_duplicate_names() {
        let result = ToolRegistry::new(vec![Arc::new(Echo("same")), Arc::new(Echo("same"))]);
        assert!(matches!(result, Err(ToolError::Duplicate(name)) if name == "same"));
    }

    #[tokio::test]
    async fn registry_reports_missing_names() {
        let registry = ToolRegistry::new(vec![]).unwrap();
        assert!(
            matches!(registry.call("missing", serde_json::Value::Null).await, Err(ToolError::NotFound(name)) if name == "missing")
        );
    }
}
