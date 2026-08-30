use crate::{
    config::QdrantConfig,
    domain::{stable_prompt_id, PromptDiscovery, PromptRecord},
    ports::{PromptRepository, StorageError},
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
#[async_trait::async_trait]
impl PromptRepository for QdrantPromptRepository {
    async fn initialize(&self) -> Result<(), StorageError> {
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
    async fn insert_prompt(&self, text: &str) -> Result<PromptRecord, StorageError> {
        let record = PromptRecord {
            id: stable_prompt_id(text),
            text: text.trim().into(),
        };
        self.request(self.client.put(self.url(&format!("collections/{}/points",self.collection))).json(&json!({"points":[{"id":record.id,"vector":[0.0],"payload":{"prompt_text":record.text,"prompt_kind":"candidate"}}]}))).await?;
        Ok(record)
    }
    async fn discover(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<PromptDiscovery>, StorageError> {
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
                Some(PromptRecord { id, text })
            })
            .collect::<Vec<_>>();
        Ok(ranking::rank(query, records, limit))
    }
}
