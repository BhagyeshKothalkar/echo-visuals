use crate::{
    config::QdrantConfig,
    domain::{SkillDiscovery, SkillRecord},
    ports::{StorageError, Tool, ToolError},
    ranking,
};
use reqwest::Client;
use serde_json::{json, Value};

fn collection_config(dimension: usize) -> Value {
    json!({"vectors": {"positive": {"size": dimension, "distance": "Cosine"}, "negative": {"size": dimension, "distance": "Cosine"}}})
}

fn embed(source: &str, dimension: usize) -> Vec<f32> {
    let mut vector = vec![0.0; dimension];
    if dimension == 0 {
        return vector;
    }
    for (index, byte) in source.bytes().enumerate() {
        let bucket = index % dimension;
        vector[bucket] += (byte as f32 / 255.0) * 2.0 - 1.0;
    }
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut vector {
            *value /= norm;
        }
    }
    vector
}

fn upsert_payload(skill: &SkillRecord, dimension: usize) -> Value {
    json!({"points": [{"id": skill.skill_id, "vector": {"positive": embed(&skill.positive_vector_source(), dimension), "negative": embed(&skill.negative_vector_source(), dimension)}, "payload": skill}]})
}

pub struct QdrantPromptRepository {
    client: Client,
    base: String,
    collection: String,
    embedding_dimension: usize,
    seed: bool,
}

impl QdrantPromptRepository {
    pub fn new(config: &QdrantConfig) -> Self {
        Self {
            client: Client::new(),
            base: config.url.trim_end_matches('/').replace(":6334", ":6333"),
            collection: config.collection.clone(),
            embedding_dimension: config.embedding_dimension,
            seed: config.seed,
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
                    .json(&collection_config(self.embedding_dimension)),
            )
            .await?;
        }
        self.request(
            self.client
                .put(self.url(&format!("collections/{}/index", self.collection)))
                .json(&json!({"field_name": "retrieval.keywords", "field_schema": "keyword"})),
        )
        .await?;
        if self.seed {
            for skill in crate::seeds::seed_skills() {
                self.insert_skill(&skill).await?;
            }
        }
        Ok(())
    }
    pub async fn insert_skill(&self, skill: &SkillRecord) -> Result<(), StorageError> {
        self.request(
            self.client
                .put(self.url(&format!("collections/{}/points", self.collection)))
                .json(&upsert_payload(skill, self.embedding_dimension)),
        )
        .await?;
        Ok(())
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
                    .json(&json!({"limit": 1000, "with_payload": true, "with_vector": false})),
            )
            .await?;
        let records = body["result"]["points"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|p| serde_json::from_value::<SkillRecord>(p.get("payload")?.clone()).ok())
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
    async fn execute(&self, input: Value) -> Result<Value, ToolError> {
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
    async fn execute(&self, input: Value) -> Result<Value, ToolError> {
        let skill: SkillRecord =
            serde_json::from_value(input).map_err(|source| ToolError::InvalidInput {
                tool: self.name().into(),
                source,
            })?;
        self.0
            .insert_skill(&skill)
            .await
            .map_err(|e| ToolError::Execution(anyhow::Error::new(e)))?;
        serde_json::to_value(skill).map_err(|e| ToolError::Execution(anyhow::Error::new(e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{Knowledge, Lifecycle, Retrieval, Usage};

    fn skill() -> SkillRecord {
        SkillRecord {
            skill_id: uuid::Uuid::nil(),
            name: "n".into(),
            description: "d".into(),
            knowledge: Knowledge {
                core: "c".into(),
                principles: vec![],
                procedures: vec![],
                failure_modes: vec![],
                examples: vec![],
            },
            usage: Usage {
                when_to_use: "use".into(),
                when_not_to_use: "avoid".into(),
                signals: vec!["signal".into()],
                anti_signals: vec!["anti".into()],
            },
            retrieval: Retrieval {
                keywords: vec!["key".into()],
            },
            lifecycle: Lifecycle {
                version: "2".into(),
                status: "active".into(),
                source: "test".into(),
                confidence: 1.0,
            },
        }
    }

    #[test]
    fn collection_config_has_named_cosine_vectors() {
        let value = collection_config(4);
        assert_eq!(value["vectors"]["positive"]["size"], 4);
        assert_eq!(value["vectors"]["positive"]["distance"], "Cosine");
        assert_eq!(value["vectors"]["negative"]["distance"], "Cosine");
    }

    #[test]
    fn upsert_contains_both_vectors_and_complete_v2_payload() {
        let value = upsert_payload(&skill(), 2);
        assert_eq!(
            value["points"][0]["vector"]["positive"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            value["points"][0]["vector"]["negative"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        assert!(value["points"][0]["payload"]["skill_id"].is_string());
        assert!(value["points"][0]["payload"]["text"].is_null());
    }

    #[test]
    fn embeddings_depend_on_their_source_text() {
        assert_ne!(embed("use", 3), embed("avoid", 3));
    }

    #[test]
    fn stored_payload_deserializes_and_projects_for_search() {
        let stored = serde_json::to_value(skill()).unwrap();
        let result: SkillRecord = serde_json::from_value(stored).unwrap();
        let projected = serde_json::to_value(result.model_projection()).unwrap();
        assert_eq!(projected.as_object().unwrap().len(), 3);
    }
}
