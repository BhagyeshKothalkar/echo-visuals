use crate::{
    config::RedisConfig,
    domain::{CandidatePrompt, FeedbackExample, FeedbackGrade},
    ports::{StorageError, Tool, ToolError},
};
use redis::{aio::ConnectionManager, AsyncCommands};
pub struct RedisFeedbackStore {
    connection: tokio::sync::Mutex<ConnectionManager>,
    prefix: String,
}
impl RedisFeedbackStore {
    pub async fn new(config: &RedisConfig) -> Result<Self, StorageError> {
        let client = redis::Client::open(config.url.as_str())
            .map_err(|e| StorageError::Backend(anyhow::Error::new(e)))?;
        let manager = client
            .get_connection_manager()
            .await
            .map_err(|e| StorageError::Backend(anyhow::Error::new(e)))?;
        Ok(Self {
            connection: tokio::sync::Mutex::new(manager),
            prefix: config.key_prefix.clone(),
        })
    }
    fn index(&self, g: FeedbackGrade) -> String {
        format!(
            "{}:{}",
            self.prefix,
            match g {
                FeedbackGrade::Positive => "positive",
                FeedbackGrade::Negative => "negative",
            }
        )
    }
    fn item(&self, id: uuid::Uuid) -> String {
        format!("{}:item:{}", self.prefix, id)
    }
}
impl RedisFeedbackStore {
    pub async fn record_feedback(
        &self,
        candidate: &CandidatePrompt,
        grade: FeedbackGrade,
    ) -> Result<(), StorageError> {
        let mut c = self.connection.lock().await;
        let seq: i64 = c
            .incr(format!("{}:sequence", self.prefix), 1)
            .await
            .map_err(|e| StorageError::Backend(anyhow::Error::new(e)))?;
        let id = candidate.record.id.to_string();
        let _: () = c
            .hset_multiple(
                self.item(candidate.record.id),
                &[
                    ("qdrant_id", id.clone()),
                    ("text", candidate.record.text.clone()),
                    ("grade", format!("{:?}", grade)),
                ],
            )
            .await
            .map_err(|e| StorageError::Backend(anyhow::Error::new(e)))?;
        let _: () = c
            .zadd(self.index(grade), id, seq)
            .await
            .map_err(|e| StorageError::Backend(anyhow::Error::new(e)))?;
        Ok(())
    }
    pub async fn top_examples(
        &self,
        grade: FeedbackGrade,
        limit: usize,
    ) -> Result<Vec<FeedbackExample>, StorageError> {
        let mut c = self.connection.lock().await;
        let ids: Vec<String> = c
            .zrevrange(self.index(grade), 0, limit.saturating_sub(1) as isize)
            .await
            .map_err(|e| StorageError::Backend(anyhow::Error::new(e)))?;
        let mut out = Vec::new();
        for id in ids {
            let text: Option<String> = c
                .hget(
                    self.item(id.parse().map_err(|_| StorageError::Invalid(id.clone()))?),
                    "text",
                )
                .await
                .map_err(|e| StorageError::Backend(anyhow::Error::new(e)))?;
            if let Some(text) = text {
                out.push(FeedbackExample {
                    id: id.parse().map_err(|_| StorageError::Invalid(id))?,
                    text,
                });
            }
        }
        Ok(out)
    }
}

pub struct GetFeedbackTool(pub std::sync::Arc<RedisFeedbackStore>);

#[async_trait::async_trait]
impl Tool for GetFeedbackTool {
    fn name(&self) -> &'static str {
        "get_feedback"
    }

    async fn execute(&self, input: serde_json::Value) -> Result<serde_json::Value, ToolError> {
        #[derive(serde::Deserialize)]
        struct Request {
            grade: FeedbackGrade,
            limit: usize,
        }
        let request: Request =
            serde_json::from_value(input).map_err(|source| ToolError::InvalidInput {
                tool: self.name().into(),
                source,
            })?;
        serde_json::to_value(
            self.0
                .top_examples(request.grade, request.limit)
                .await
                .map_err(|e| ToolError::Execution(anyhow::Error::new(e)))?,
        )
        .map_err(|e| ToolError::Execution(anyhow::Error::new(e)))
    }
}

pub struct RecordFeedbackTool(pub std::sync::Arc<RedisFeedbackStore>);

#[async_trait::async_trait]
impl Tool for RecordFeedbackTool {
    fn name(&self) -> &'static str {
        "record_feedback"
    }

    async fn execute(&self, input: serde_json::Value) -> Result<serde_json::Value, ToolError> {
        #[derive(serde::Deserialize)]
        struct Request {
            id: uuid::Uuid,
            text: String,
            grade: FeedbackGrade,
        }
        let request: Request =
            serde_json::from_value(input).map_err(|source| ToolError::InvalidInput {
                tool: self.name().into(),
                source,
            })?;
        let candidate = CandidatePrompt {
            record: crate::domain::PromptRecord {
                id: request.id,
                text: request.text,
            },
        };
        self.0
            .record_feedback(&candidate, request.grade)
            .await
            .map_err(|e| ToolError::Execution(anyhow::Error::new(e)))?;
        Ok(serde_json::json!({}))
    }
}
