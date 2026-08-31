use crate::{
    config::QdrantConfig,
    domain::{stable_prompt_id, SkillDiscovery, SkillRecord},
    ports::{StorageError, Tool, ToolError},
    ranking,
};
use reqwest::Client;
use serde_json::{json, Value};

pub struct QdrantPromptRepository {
    client: Client,
    base: String,
    collection: String,
}
impl QdrantPromptRepository {
    pub fn new(config: &QdrantConfig) -> Self {
        Self {
            client: Client::new(),
            base: config.url.trim_end_matches("/").replace(":6334", ":6333"),
            collection: config.collection.clone(),
        }
    }
    fn url(&self, path: &str) -> String {
        format!("{}/{}", self.base, path.trim_start_matches('/'))
    }
    async fn request(&self, builder: reqwest::RequestBuilder) -> Result<Value, StorageError> {
        let response = builder
            .send()
            .await
            .map_err(|e| StorageError::Backend(anyhow::Error::new(e)))?;
        let status = response.status();
        let body = response
            .json::<Value>()
            .await
            .map_err(|e| StorageError::Backend(anyhow::Error::new(e)))?;
        if !status.is_success() {
            return Err(StorageError::Backend(anyhow::anyhow!(
                "Qdrant returned {}: {}",
                status,
                body
            )));
        }
        Ok(body)
    }
}
impl QdrantPromptRepository {
    pub async fn initialize(&self) -> Result<(), StorageError> {
        let get = self
            .client
            .get(self.url(&format!("collections/{}", self.collection)))
            .send()
            .await
            .map_err(|e| StorageError::Backend(anyhow::Error::new(e)))?;
        if get.status() == reqwest::StatusCode::NOT_FOUND {
            self.request(
                self.client
                    .put(self.url(&format!("collections/{}", self.collection)))
                    .json(&json!({"vectors":{"size":1,"distance":"Dot"}})),
            )
            .await?;
        }
        self.request(
            self.client
                .put(self.url(&format!("collections/{}/index", self.collection)))
                .json(&json!({"field_name":"prompt_text","field_schema":"text"})),
        )
        .await?;
        for seed in crate::seeds::SEED_PROMPTS {
            self.insert_prompt(seed).await?;
        }
        Ok(())
    }
    pub async fn insert_prompt(&self, text: &str) -> Result<SkillRecord, StorageError> {
        let record = SkillRecord {
            id: stable_prompt_id(text),
            text: text.trim().into(),
        };
        self.request(self.client.put(self.url(&format!("collections/{}/points",self.collection))).json(&json!({"points":[{"id":record.id,"vector":[0.0],"payload":{"prompt_text":record.text,"prompt_kind":"skill"}}]}))).await?;
        Ok(record)
    }
    pub async fn discover(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<SkillDiscovery>, StorageError> {
        let body = self
            .request(
                self.client
                    .post(self.url(&format!("collections/{}/points/scroll", self.collection)))
                    .json(&json!({"limit":1000,"with_payload":true,"with_vector":false})),
            )
            .await?;
        let records = body["result"]["points"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|p| {
                let id = p["id"].as_str()?.parse().ok()?;
                let text = p["payload"]["prompt_text"].as_str()?.to_string();
                Some(SkillRecord { id, text })
            })
            .collect::<Vec<_>>();
        Ok(ranking::rank(query, records, limit))
    }
}

pub struct SearchSkillsTool(pub std::sync::Arc<QdrantPromptRepository>);

#[async_trait::async_trait]
impl Tool for SearchSkillsTool {
    fn name(&self) -> &'static str {
        "search_skills"
    }

    async fn execute(&self, input: serde_json::Value) -> Result<serde_json::Value, ToolError> {
        #[derive(serde::Deserialize)]
        struct Request {
            query: String,
            limit: usize,
        }
        let request: Request =
            serde_json::from_value(input).map_err(|source| ToolError::InvalidInput {
                tool: self.name().into(),
                source,
            })?;
        serde_json::to_value(
            self.0
                .discover(&request.query, request.limit)
                .await
                .map_err(|e| ToolError::Execution(anyhow::Error::new(e)))?,
        )
        .map_err(|e| ToolError::Execution(anyhow::Error::new(e)))
    }
}

pub struct SaveSkillTool(pub std::sync::Arc<QdrantPromptRepository>);

#[async_trait::async_trait]
impl Tool for SaveSkillTool {
    fn name(&self) -> &'static str {
        "save_skill"
    }

    async fn execute(&self, input: serde_json::Value) -> Result<serde_json::Value, ToolError> {
        #[derive(serde::Deserialize)]
        struct Request {
            text: String,
        }
        let request: Request =
            serde_json::from_value(input).map_err(|source| ToolError::InvalidInput {
                tool: self.name().into(),
                source,
            })?;
        serde_json::to_value(
            self.0
                .insert_prompt(&request.text)
                .await
                .map_err(|e| ToolError::Execution(anyhow::Error::new(e)))?,
        )
        .map_err(|e| ToolError::Execution(anyhow::Error::new(e)))
    }
}
