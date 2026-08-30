# Agentic Prompt Improver

Rust MVP for iteratively improving a target prompt with Qdrant discovery, Redis feedback, and a Rig OpenAI-compatible LLM client.

## Run

1. Start services: `docker compose up -d`.
2. Copy `config.example.toml` to `config.toml` and edit public settings.
3. Copy `.env.example` to `.env` and set `LLM_API_KEY`.
4. Seed Qdrant: `cargo run -- init` (safe to repeat).
5. Generate once: `cargo run -- run --target "Write a clear project plan"`.
6. Iterate with feedback: `cargo run -- interactive --target "Write a clear project plan"`.

The configuration file controls URLs, model, retrieval limits, logging, and prompt paths without recompiling. Credentials are environment-only. Prompt behavior is editable in `prompts/agents/` and `prompts/templates/` Markdown files.

Tests use in-memory logic and do not require Docker or an API key: `cargo test`.
