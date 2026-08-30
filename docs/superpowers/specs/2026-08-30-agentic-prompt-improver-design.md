# Agentic Prompt Improver Design

## Goal

Build a scaffolded asynchronous Rust MVP in which one LLM agent iteratively improves a user-provided target prompt using prior positive and negative feedback, Redis-backed examples, and Qdrant-backed prompt discovery.

## Scope and decisions

- The MVP is one binary and one agent implementation, with interfaces designed for additional agents, ranking strategies, and LLM providers.
- Rig's OpenAI-compatible client is the default LLM implementation. Its API key and optional organization/provider credentials come from environment variables or `.env`; its model, base URL, and generation settings come from TOML.
- Qdrant is the durable prompt store and discovery source. The MVP uses deterministic lexical token-overlap ranking over prompt records loaded through the Qdrant client; the ranking implementation is isolated so an embedding or hybrid strategy can be added later.
- Redis is a separate feedback cache. Positive and negative examples use separate namespaces and retain each prompt's stable Qdrant ID, text, grade, and timestamps.
- Initialization creates the Qdrant collection and payload indexes as needed, then upserts the seed prompts. Stable UUIDv5 IDs derived from prompt text make repeated initialization idempotent.
- The harness is the deterministic policy boundary. Every iteration retrieves exactly up to two positive examples, two negative examples, and two Qdrant discoveries, in stable order, before calling the LLM.
- A CLI supports initialization, one-shot generation, and an interactive feedback loop. The core flow remains independently testable without live services.

## Architecture

```text
CLI -> Harness -> Agent
                 |-> RedisStore (positive/negative feedback)
                 |-> PromptRepository (Qdrant storage + discovery)
                 `-> LlmProvider (Rig OpenAI-compatible implementation)
```

### Configuration

`Config` loads defaults, then a user-selected `config.toml` (with checked-in `config.example.toml` as the safe template), then environment variables for secret/credential fields only. TOML contains all non-secret, app-specific/public settings:

- Qdrant URL, collection, seed behavior, and tunables.
- Redis URL, key prefix, and tunables.
- OpenAI-compatible base URL, model, generation settings, and prompt asset paths.
- Logging, retrieval limits, and other application tunables.

Environment variables are reserved for secrets and credentials, such as `LLM_API_KEY` and optional organization or provider token values. `dotenvy` loads these secret values from a local `.env` for development. Configuration is strongly typed and validated at startup; secrets are not required in TOML and no secret is committed. For public settings, the effective precedence is defaults < TOML; for credential settings, environment values override any absent or optional configured value. App behavior, including retrieval limits and prompt paths, can change through TOML without recompiling.

### Prompt templates

All reusable prompt content is stored as editable Markdown files at the project root under `prompts/`, with agent/system prompts in `prompts/agents/` and user-message/templates in `prompts/templates/`. `src/prompts/` contains only loader, typed context, and rendering logic. Prompt paths are configurable through TOML where practical. The user message is rendered from a typed `PromptContext` containing the target and six labeled examples. The LLM contract requires a plain candidate prompt response, with trimming and empty-output validation at the adapter boundary. Editing a Markdown prompt asset changes behavior without changing Rust source or recompiling.

### Qdrant repository

`PromptRepository` owns collection lifecycle, seed upserts, prompt insertion, and discovery. Its public model is independent of Qdrant wire types:

- `PromptRecord { id, text, metadata }`
- `PromptDiscovery { record, score }`
- `insert_prompt(text) -> PromptRecord`
- `discover(query, limit) -> Vec<PromptDiscovery>`
- `initialize() -> ()`

The MVP Qdrant implementation scrolls prompt records and scores them with normalized token overlap (lowercase Unicode alphanumeric tokens, query-token coverage, then stable ID tie-break). It excludes no records by default; the harness can later pass filters or a strategy can evolve. Qdrant payload contains at minimum `prompt_text` and `prompt_kind`, with a text payload index created during initialization.

### Redis feedback store

`FeedbackStore` exposes `record_feedback` and `top_examples`. The Redis implementation stores per-grade sorted indexes and per-prompt hashes. The sorted member is the Qdrant ID; the hash retains prompt text, grade, and timestamps. Scores are deterministic insertion sequence values, and retrieval sorts by newest sequence then ID. The store can reuse a Qdrant ID directly and never re-embeds a Redis example.

### LLM abstraction

`LlmProvider` is an async trait with one operation that accepts `PromptContext` and returns a candidate string. `RigOpenAiProvider` owns Rig's OpenAI-compatible client and maps configuration to a completion request. A future provider implements the same trait without changing `Agent` or `Harness`.

### Agent and harness

`Agent` keeps the target prompt and performs one generation iteration through injected repository, feedback store, and LLM traits:

1. Ask the feedback store for two positive and two negative examples.
2. Ask the repository for two discoveries using the target.
3. Build the typed prompt context and call the LLM.
4. Insert the candidate into Qdrant.
5. Return the candidate record.

The feedback operation is separate and records the returned candidate's Qdrant ID plus its text. This preserves the loop boundary and allows a caller to grade after displaying the result. The harness has no random sampling, wall-clock ordering, or hidden retries; ordering and limits are explicit constants.

## Error handling and observability

All I/O is async. Typed errors are lightweight and limited to meaningful module/API boundaries, using a small number of top-level/domain categories with source errors preserved where useful. There are no per-operation wrapper types or broad error-conversion layer. `tracing` logs initialization, iteration IDs, selected example counts, and feedback grades; secrets and full API keys are never logged. CLI errors produce a non-zero exit code.

## Testing strategy

- Unit tests for tokenization and deterministic lexical ranking.
- Repository contract tests using an in-memory fake for seed idempotency and stable IDs.
- Feedback store tests using an in-memory fake for grade separation and Qdrant-ID reuse.
- Harness/agent tests using fakes to assert exact retrieval limits, context labels, candidate persistence, and feedback persistence.
- Configuration tests for required/optional environment values.
- Configuration precedence and TOML validation tests, including the guarantee that secrets are not required in TOML.
- Prompt-loader tests proving Markdown assets render without Rust-source prompt text.
- Live integration testing is documented as an optional Docker Compose check and is not required for default `cargo test`.

## Operational files

- `docker-compose.yml` runs Qdrant and Redis with persistent named volumes.
- `config.example.toml` documents non-secret application settings and defaults.
- `prompts/agents/` and `prompts/templates/` contain the editable Markdown prompt assets.
- `.env.example` documents secret/credential variable names only.
- `README.md` documents prerequisites, TOML and `.env` setup, startup, initialization, one-shot invocation, and the feedback loop.
- `Cargo.toml` pins the async/runtime, Rig, Qdrant, Redis, serde, config/error, CLI, logging, and test dependencies.

## Acceptance criteria

1. `cargo test` passes without external services.
2. `cargo fmt --check` and `cargo clippy --all-targets --all-features -- -D warnings` pass.
3. Defaults < TOML < environment-secret precedence is tested, and app behavior can be changed through TOML without recompiling.
4. Secrets are not required in TOML and no secret is committed.
5. With Docker Compose and valid LLM credentials, initialization can be repeated without duplicate seed prompts.
6. Each generation stores a stable-ID Qdrant prompt and returns that ID to the CLI.
7. Positive and negative feedback are stored separately in Redis and retain the Qdrant ID.
8. Each subsequent iteration supplies the target plus exactly the configured six example slots (or fewer only when stores contain fewer records) in deterministic order.
9. Prompt content can be modified by editing Markdown files under `prompts/` without changing Rust source.
