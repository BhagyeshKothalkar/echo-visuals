# Agentic Prompt Improver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan inline, task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a modular async Rust MVP that improves a target prompt through Qdrant discovery, Redis feedback, and an OpenAI-compatible Rig LLM provider.

**Architecture:** Keep a small library with focused modules for configuration, prompt assets, domain ports, Qdrant, Redis, LLM, agent, and harness. The CLI wires those modules together. Qdrant stores stable-ID prompt records and provides the source records for deterministic lexical discovery; Redis stores separate positive/negative feedback indexes; traits keep storage, ranking, and LLM implementations replaceable.

**Tech Stack:** Rust 2021, Tokio, Rig OpenAI-compatible client, Qdrant client, Redis client, Serde/TOML, Clap, Dotenvy, Thiserror, Tracing, async-trait.

**Spec:** `docs/superpowers/specs/2026-08-30-agentic-prompt-improver-design.md`

## Global Constraints

- Public application settings live in TOML; secrets and credentials live only in environment variables or local `.env`.
- Public setting precedence is defaults < TOML; credential environment values override optional credential settings.
- Prompt content is Markdown under root `prompts/`; `src/prompts/` contains loader/types/rendering only.
- Qdrant seed initialization is idempotent and prompt IDs are stable.
- Harness retrieval is deterministic and requests two positive, two negative, and two Qdrant examples.
- Errors use small meaningful boundary categories; avoid per-operation wrapper enums.
- Default tests run without Qdrant, Redis, or LLM services.

---

### Step 1: Implement the complete core library with test-first coverage

**Files:**
- Create: `Cargo.toml`, `src/lib.rs`, `src/config.rs`, `src/domain.rs`, `src/ports.rs`, `src/ranking.rs`
- Create: `src/prompts/mod.rs`, `src/prompts/loader.rs`, `src/prompts/render.rs`
- Create: `src/qdrant.rs`, `src/redis_store.rs`, `src/llm.rs`, `src/agent.rs`, `src/harness.rs`, `src/seeds.rs`
- Create: `prompts/agents/prompt-improver-system.md`, `prompts/templates/candidate-request.md`
- Create: `config.example.toml`, `.env.example`
- Test: unit and module tests colocated with their implementation modules

**Interfaces:**
- `Config::from_sources(path: Option<&Path>) -> Result<Config, ConfigError>` loads defaults, TOML public settings, and credential environment values; validates all startup tunables.
- `PromptAssets::load(&PromptAssetConfig) -> Result<PromptAssets, PromptError>` and `PromptAssets::render(&PromptContext) -> Result<RenderedPrompt, PromptError>` load root Markdown at runtime.
- `PromptRepository` exposes `initialize`, `insert_prompt`, and `discover`; `FeedbackStore` exposes `record_feedback` and `top_examples`; `LlmProvider` accepts rendered prompt context and returns a candidate.
- `QdrantPromptRepository` implements idempotent stable-ID seed upserts and scroll-plus-lexical discovery over payload prompt text.
- `RedisFeedbackStore` implements separate positive/negative indexes retaining each Qdrant ID, text, grade, and insertion sequence.
- `Agent::iterate() -> Result<CandidatePrompt, AgentError>` retrieves the configured examples, calls the LLM, inserts the candidate, and returns its stable Qdrant ID; `Agent::grade(...)` persists feedback.
- `Harness` fixes default retrieval limits at positive=2, negative=2, discoveries=2 and passes them deterministically to the agent.
- `RigOpenAiProvider` uses the configured custom OpenAI-compatible base URL/model/generation settings and credential environment values.

- [ ] **Write the failing core tests first.** Cover config defaults/TOML precedence/validation, Markdown loading/rendering, lexical token-overlap ordering and limits, stable IDs and seed idempotency via an in-memory repository contract, Redis grade separation and Qdrant-ID retention via an in-memory feedback contract, and one agent iteration with recording fakes asserting exact retrieval limits, context labels, candidate insertion, and grading.
- [ ] **Run the focused tests and observe RED.** Use `cargo test config`, `cargo test prompts`, `cargo test ranking`, and the agent/storage test filters; confirm failures are caused by missing implementation rather than test mistakes.
- [ ] **Implement the minimal domain and configuration layer.** Add the manifest, typed settings, defaults/TOML loading, credential-only environment loading, lightweight boundary errors, stable UUIDv5 IDs, domain records, and async traits.
- [ ] **Implement Markdown prompt loading/rendering and deterministic lexical ranking.** Keep reusable prose exclusively in the two Markdown files; Rust should only load, validate placeholders, tokenize, score, and format typed examples.
- [ ] **Implement Qdrant and Redis adapters.** Qdrant should create the collection/index, upsert seed payloads by stable ID, insert candidates, scroll prompt payloads, and rank them lexically. Redis should use separate grade keys and persist the Qdrant ID in each feedback record. Map backend failures to small storage/LLM boundary errors.
- [ ] **Implement the agent, harness, and Rig adapter.** Keep orchestration dependent only on ports; use Rig for the OpenAI-compatible request, trim/reject empty output, and leave credentials out of logs.
- [ ] **Run all core tests and `cargo fmt --check`.** Fix production code until tests pass; preserve tests that run without external services. Commit `feat: implement agentic prompt improver core`.

### Step 2: Add the thin CLI/runtime surface and verify the MVP

**Files:**
- Create: `src/main.rs`, `docker-compose.yml`, `README.md`
- Modify: `config.example.toml`, `.env.example` if documentation needs refinement
- Test: CLI parsing tests in `src/main.rs` or `src/cli.rs` if extraction keeps parsing clearer

**Interfaces:**
- CLI commands are `init`, `run --target <text>`, and `interactive --target <text>`; each accepts `--config <path>`.
- `init` initializes Qdrant and seeds prompts; `run` generates one candidate and prints text plus stable Qdrant ID; `interactive` repeats generation and accepts only `positive` or `negative` feedback.

- [ ] **Write failing CLI parsing tests.** Assert command/config-path parsing and invalid grade rejection without constructing live services.
- [ ] **Run the CLI tests and observe RED.** Confirm parsing tests fail because the CLI surface is not implemented.
- [ ] **Implement the thin runtime wiring.** Load `Config`, initialize tracing, construct Qdrant/Redis/Rig adapters, initialize Qdrant for `init`, and wire the iteration/feedback loop for `run` and `interactive`; return non-zero errors without logging secrets.
- [ ] **Add Docker Compose and examples.** Run Qdrant and Redis with named persistent volumes; document TOML configuration, `.env` credentials, Docker startup, initialization, one-shot invocation, interactive feedback, and editing Markdown prompt assets without recompilation.
- [ ] **Run the complete verification suite.** Execute `cargo fmt --check`, `cargo test`, `cargo clippy --all-targets --all-features -- -D warnings`, `cargo check`, and `git diff --check`. Confirm no secrets are tracked, all prompt prose is under root `prompts/`, and TOML can alter retrieval/runtime behavior without recompiling.
- [ ] **Commit `feat: add cli and local services`** if the verification suite is clean, then report the worktree, commands, and test results.
