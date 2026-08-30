use crate::{agent::Agent, config::RetrievalConfig, domain::CandidatePrompt};
#[derive(Debug, thiserror::Error)]
pub enum HarnessError {
    #[error(transparent)]
    Agent(#[from] crate::agent::AgentError),
}
#[derive(Clone, Debug)]
pub struct Harness {
    pub retrieval: RetrievalConfig,
}
impl Harness {
    pub fn new(retrieval: RetrievalConfig) -> Self {
        Self { retrieval }
    }
    pub async fn run_iteration(&self, agent: &Agent) -> Result<CandidatePrompt, HarnessError> {
        Ok(agent.iterate(&self.retrieval).await?)
    }
}
