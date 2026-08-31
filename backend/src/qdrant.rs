use crate::{
    config::QdrantConfig,
    domain::{SkillDiscovery, SkillRecord},
    ports::{Embedder, SkillStore, StorageError, Tool, ToolError},
};
use reqwest::Client;
use serde_json::{json, Value};

fn collection_config(dimension: usize) -> Value {
    json!({
        "vectors": {
            "positive": {
                "size": dimension,
                "distance": "Cosine"
            },
            "negative": {
                "size": dimension,
                "distance": "Cosine"
            }
        }
    })
}

fn upsert_payload(skill: &SkillRecord, positive: Vec<f32>, negative: Vec<f32>) -> Value {
    json!({
        "points": [{
            "id": skill.skill_id,
            "vector": {
                "positive": positive,
                "negative": negative
            },
            "payload": skill
        }]
    })
}

fn projected_search_results(body: &Value) -> Vec<SkillDiscovery> {
    body["result"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|point| serde_json::from_value(point.get("payload")?.clone()).ok())
        .collect()
}

fn exact_skill_from_response(body: &Value, id: uuid::Uuid) -> Result<SkillRecord, StorageError> {
    body["result"]
        .as_array()
        .into_iter()
        .flatten()
        .find_map(|point| serde_json::from_value::<SkillRecord>(point.get("payload")?.clone()).ok())
        .filter(|skill| skill.skill_id == id)
        .ok_or_else(|| StorageError::Invalid(format!("skill not found: {id}")))
}

pub fn search_request(vector: Vec<f32>, limit: usize) -> Value {
    json!({
        "vector": {
            "name": "positive",
            "vector": vector
        },
        "limit": limit,
        "with_payload": true,
        "with_vector": false
    })
}

pub fn retrieve_request(id: uuid::Uuid) -> Value {
    json!({
        "ids": [id],
        "with_payload": true,
        "with_vector": false
    })
}

pub struct QwenEmbeddingEmbedder {
    client: Client,
    endpoint: String,
    model: String,
    device: String,
    dimension: usize,
    batch_size: usize,
    normalize: bool,
}

impl QwenEmbeddingEmbedder {
    pub fn new(config: &crate::config::EmbeddingConfig) -> Self {
        Self {
            client: Client::new(),
            endpoint: config.endpoint.clone(),
            model: config.model.clone(),
            device: config.device.clone(),
            dimension: config.dimension,
            batch_size: config.batch_size.max(1),
            normalize: config.normalize,
        }
    }

    async fn request(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, StorageError> {
        let body = self
            .client
            .post(&self.endpoint)
            .json(&json!({
                "model": self.model,
                "input": texts,
                "encoding_format": "float",
                "normalize": self.normalize,
                "device": self.device
            }))
            .send()
            .await
            .map_err(|e| StorageError::Backend(anyhow::Error::new(e)))?;

        let status = body.status();

        let value: Value = body
            .json()
            .await
            .map_err(|e| StorageError::Backend(anyhow::Error::new(e)))?;

        if !status.is_success() {
            return Err(StorageError::Backend(anyhow::anyhow!(
                "embedding server returned {}: {}",
                status,
                value
            )));
        }

        let mut out: Vec<(usize, Vec<f32>)> = value["data"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|item| {
                (
                    item["index"].as_u64().unwrap_or(0) as usize,
                    serde_json::from_value(item["embedding"].clone()).unwrap_or_default(),
                )
            })
            .collect();

        out.sort_by_key(|(index, _)| *index);

        let vectors = out
            .into_iter()
            .map(|(_, vector)| vector)
            .collect::<Vec<_>>();

        validate_dimensions(&vectors, self.dimension)?;

        Ok(vectors)
    }
}

fn validate_dimensions(vectors: &[Vec<f32>], dimension: usize) -> Result<(), StorageError> {
    if vectors.iter().any(|vector| vector.len() != dimension) {
        return Err(StorageError::Invalid(format!(
            "embedding dimension must be {dimension}"
        )));
    }

    Ok(())
}

#[async_trait::async_trait]
impl Embedder for QwenEmbeddingEmbedder {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, StorageError> {
        let mut result = Vec::with_capacity(texts.len());

        for batch in texts.chunks(self.batch_size) {
            result.extend(self.request(batch).await?);
        }

        Ok(result)
    }
}

pub struct QdrantPromptRepository {
    client: Client,
    base: String,
    collection: String,
    embedding_dimension: usize,
    seed: bool,
    embedder: std::sync::Arc<dyn Embedder>,
}

impl QdrantPromptRepository {
    pub fn new(config: &QdrantConfig, embedder: std::sync::Arc<dyn Embedder>) -> Self {
        Self {
            client: Client::new(),
            base: config.url.trim_end_matches('/').replace(":6334", ":6333"),
            collection: config.collection.clone(),
            embedding_dimension: config.embedding_dimension,
            seed: config.seed,
            embedder,
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

    fn validate_collection_schema(&self, value: &Value) -> Result<(), StorageError> {
        let vectors = &value["result"]["config"]["params"]["vectors"];

        let positive_size = vectors["positive"]["size"].as_u64();
        let positive_distance = vectors["positive"]["distance"].as_str();

        let negative_size = vectors["negative"]["size"].as_u64();
        let negative_distance = vectors["negative"]["distance"].as_str();

        let expected_dimension = self.embedding_dimension as u64;

        let schema_matches = positive_size == Some(expected_dimension)
            && negative_size == Some(expected_dimension)
            && positive_distance == Some("Cosine")
            && negative_distance == Some("Cosine");

        if !schema_matches {
            return Err(StorageError::Invalid(format!(
                "Qdrant collection '{}' has an incompatible schema; \
                 expected named positive/negative Cosine vectors with \
                 dimension {}, recreate the collection with the V2 schema",
                self.collection, self.embedding_dimension
            )));
        }

        Ok(())
    }

    pub async fn initialize(&self) -> Result<(), StorageError> {
        let get = self
            .client
            .get(self.url(&format!("collections/{}", self.collection)))
            .send()
            .await
            .map_err(|e| StorageError::Backend(anyhow::Error::new(e)))?;

        let get_status = get.status();

        match get_status {
            reqwest::StatusCode::OK => {
                let value = get
                    .json::<Value>()
                    .await
                    .map_err(|e| StorageError::Backend(anyhow::Error::new(e)))?;

                self.validate_collection_schema(&value)?;
            }

            reqwest::StatusCode::NOT_FOUND => {
                self.request(
                    self.client
                        .put(self.url(&format!("collections/{}", self.collection)))
                        .json(&collection_config(self.embedding_dimension)),
                )
                .await?;
            }

            status => {
                let body = get
                    .text()
                    .await
                    .unwrap_or_else(|_| "<unreadable body>".into());

                return Err(StorageError::Backend(anyhow::anyhow!(
                    "Qdrant collection check returned {}: {}",
                    status,
                    body
                )));
            }
        }

        self.request(
            self.client
                .put(self.url(&format!("collections/{}/index", self.collection)))
                .json(&json!({
                    "field_name": "retrieval.keywords",
                    "field_schema": "keyword"
                })),
        )
        .await?;

        if self.seed {
            for skill in crate::seeds::seed_skills() {
                self.insert_skill(&skill).await?;
            }
        }

        self.reindex_existing().await?;

        Ok(())
    }

    async fn reindex_existing(&self) -> Result<(), StorageError> {
        let body = self
            .request(
                self.client
                    .post(self.url(&format!("collections/{}/points/scroll", self.collection)))
                    .json(&json!({
                        "limit": 1000,
                        "with_payload": true,
                        "with_vector": false
                    })),
            )
            .await?;

        let points = body["result"]["points"].as_array().ok_or_else(|| {
            StorageError::Invalid("Qdrant scroll response missing result.points".into())
        })?;

        for point in points {
            let payload = point.get("payload").ok_or_else(|| {
                StorageError::Invalid(format!(
                    "Qdrant point {:?} is missing payload",
                    point.get("id")
                ))
            })?;

            let skill: SkillRecord = serde_json::from_value(payload.clone()).map_err(|e| {
                StorageError::Invalid(format!(
                    "invalid V2 SkillRecord payload in Qdrant point {:?}: {}",
                    point.get("id"),
                    e
                ))
            })?;

            self.insert_skill(&skill).await?;
        }

        Ok(())
    }

    pub async fn insert_skill(&self, skill: &SkillRecord) -> Result<(), StorageError> {
        let vectors = self
            .embedder
            .embed(&[
                skill.positive_vector_source(),
                skill.negative_vector_source(),
            ])
            .await?;

        if vectors.len() != 2 {
            return Err(StorageError::Invalid(
                "embedder returned an unexpected vector count".into(),
            ));
        }

        self.request(
            self.client
                .put(self.url(&format!("collections/{}/points", self.collection)))
                .json(&upsert_payload(
                    skill,
                    vectors[0].clone(),
                    vectors[1].clone(),
                )),
        )
        .await?;

        Ok(())
    }

    pub async fn discover(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<SkillDiscovery>, StorageError> {
        let vector = self.embedder.embed(&[query.to_string()]).await?.remove(0);

        let body = self
            .request(
                self.client
                    .post(self.url(&format!("collections/{}/points/search", self.collection)))
                    .json(&search_request(vector, limit)),
            )
            .await?;

        Ok(projected_search_results(&body))
    }

    pub async fn get_skill(&self, id: uuid::Uuid) -> Result<SkillRecord, StorageError> {
        let body = self
            .request(
                self.client
                    .post(self.url(&format!("collections/{}/points", self.collection)))
                    .json(&retrieve_request(id)),
            )
            .await?;

        exact_skill_from_response(&body, id)
    }
}

#[async_trait::async_trait]
impl SkillStore for QdrantPromptRepository {
    async fn get_skill(&self, id: uuid::Uuid) -> Result<SkillRecord, StorageError> {
        self.get_skill(id).await
    }

    async fn save_skill(&self, skill: &SkillRecord) -> Result<(), StorageError> {
        self.insert_skill(skill).await
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
        assert_eq!(value["vectors"]["negative"]["size"], 4);
        assert_eq!(value["vectors"]["negative"]["distance"], "Cosine");
    }

    #[test]
    fn vector_search_request_uses_positive_named_vector_and_limit() {
        let request = search_request(vec![0.25, 0.75], 5);

        assert_eq!(request["vector"]["name"], "positive");
        assert_eq!(request["vector"]["vector"][1], 0.75);
        assert_eq!(request["limit"], 5);
        assert_eq!(request["with_payload"], true);
    }

    #[test]
    fn exact_lookup_request_targets_only_requested_point() {
        let id = uuid::Uuid::nil();

        assert_eq!(retrieve_request(id)["ids"][0], id.to_string());
        assert_eq!(retrieve_request(id)["with_payload"], true);
    }

    #[test]
    fn search_results_and_exact_lookup_preserve_skill_id_and_missing_behavior() {
        let id = uuid::Uuid::nil();
        let payload = serde_json::to_value(skill()).unwrap();
        let body = json!({
            "result": [{
                "id": id,
                "payload": payload
            }]
        });

        assert_eq!(projected_search_results(&body)[0].skill_id, id);
        assert_eq!(exact_skill_from_response(&body, id).unwrap().skill_id, id);

        assert!(matches!(
            exact_skill_from_response(
                &body,
                uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_URL, b"missing")
            ),
            Err(StorageError::Invalid(_))
        ));
    }

    #[test]
    fn upsert_contains_both_vectors_and_complete_v2_payload() {
        let value = upsert_payload(&skill(), vec![0.0, 1.0], vec![1.0, 0.0]);

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
    fn embedding_dimensions_are_explicit() {
        assert!(validate_dimensions(&[vec![0.0; 1024]], 1024).is_ok());

        assert!(validate_dimensions(&[vec![0.0; 3]], 1024).is_err());
    }

    #[test]
    fn stored_payload_deserializes_and_projects_for_search() {
        let stored = serde_json::to_value(skill()).unwrap();

        let result: SkillRecord = serde_json::from_value(stored).unwrap();

        let projected = serde_json::to_value(result.model_projection()).unwrap();

        assert_eq!(projected.as_object().unwrap().len(), 4);
    }

    #[test]
    fn compatible_collection_schema_is_accepted() {
        let config = QdrantConfig {
            url: "http://localhost:6333".into(),
            collection: "skills".into(),
            embedding_dimension: 1024,
            seed: false,
        };

        let embedder: std::sync::Arc<dyn Embedder> = std::sync::Arc::new(
            QwenEmbeddingEmbedder::new(&crate::config::EmbeddingConfig {
                endpoint: "http://localhost:8080/v1/embeddings".into(),
                model: "test".into(),
                device: "cpu".into(),
                dimension: 1024,
                batch_size: 8,
                normalize: true,
            }),
        );

        let repository = QdrantPromptRepository::new(&config, embedder);

        let value = json!({
            "result": {
                "config": {
                    "params": {
                        "vectors": {
                            "positive": {
                                "size": 1024,
                                "distance": "Cosine"
                            },
                            "negative": {
                                "size": 1024,
                                "distance": "Cosine"
                            }
                        }
                    }
                }
            }
        });

        assert!(repository.validate_collection_schema(&value).is_ok());
    }

    #[test]
    fn incompatible_collection_schema_is_rejected() {
        let config = QdrantConfig {
            url: "http://localhost:6333".into(),
            collection: "skills".into(),
            embedding_dimension: 1024,
            seed: false,
        };

        let embedder: std::sync::Arc<dyn Embedder> = std::sync::Arc::new(
            QwenEmbeddingEmbedder::new(&crate::config::EmbeddingConfig {
                endpoint: "http://localhost:8080/v1/embeddings".into(),
                model: "test".into(),
                device: "cpu".into(),
                dimension: 1024,
                batch_size: 8,
                normalize: true,
            }),
        );

        let repository = QdrantPromptRepository::new(&config, embedder);

        let value = json!({
            "result": {
                "config": {
                    "params": {
                        "vectors": {
                            "positive": {
                                "size": 1,
                                "distance": "Cosine"
                            },
                            "negative": {
                                "size": 1,
                                "distance": "Cosine"
                            }
                        }
                    }
                }
            }
        });

        assert!(matches!(
            repository.validate_collection_schema(&value),
            Err(StorageError::Invalid(_))
        ));
    }
}
