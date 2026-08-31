use crate::{agent::Agent, domain::CandidatePrompt};
#[derive(Debug, thiserror::Error)]
pub enum HarnessError {
    #[error(transparent)]
    Agent(#[from] crate::agent::AgentError),
}
#[derive(Clone, Debug, Default)]
pub struct Harness;
impl Harness {
    pub fn new() -> Self {
        Self
    }
    pub async fn run_iteration(&self, agent: &Agent) -> Result<CandidatePrompt, HarnessError> {
        let result = agent.iterate().await?;
        Ok(CandidatePrompt {
            id: crate::domain::stable_prompt_id(&result.text),
            text: result.text,
        })
    }
}
