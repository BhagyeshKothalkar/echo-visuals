# Echo Visuals V2

This service improves an image-generation target in one bounded pass:

```text
target + optional image → Analyst (VLM) → skill IDs + surfaced feedback → Rust lookup → Optimizer (LLM) → optional Curator (LLM)
```

The Analyst owns observations, weaknesses, requirements, skill retrieval, and feedback retrieval. It may call only `search_skills` and `get_feedback`, and returns IDs plus an explicit `feedback` array rather than leaving tool results in the model context. Rust deduplicates those IDs in order and resolves each record directly from Qdrant. A missing ID stops the pass before optimization.

The Optimizer always produces the candidate prompt and decides whether knowledge is reusable enough to save. The Curator runs only for that decision, emits one canonical V2 `SkillRecord`, and Rust validates and saves it through Qdrant. The candidate is always the optimizer prompt with `stable_prompt_id(prompt)`; a saved skill is supporting knowledge, never the candidate.

There are exactly two model generations without saving and three when saving. No critic, generator loop, retry, fallback save, or recursive orchestration is used. Redis remains feedback memory: the application records feedback from the CLI, while only the Analyst can retrieve it.

Run the unchanged CLI commands:

```bash
cargo run -- init
cargo run -- run --target "..." --image ./image.png
cargo run -- interactive --target "..." --image ./image.png
```

Qdrant is V2-only: each collection must have named `positive` and `negative` Cosine vectors at the configured dimension. Startup creates a missing collection and ensures its required indexes, but does not migrate or reindex existing data. If the embedding schema changes, manually recreate the collection before starting the service. Embedding inference is local and configured in `[embedding]`; the default is `Qwen/Qwen3-Embedding-0.6B` at 1024 dimensions on CUDA through an OpenAI-compatible embeddings endpoint. `[analyst]`, `[optimizer]`, and `[curator]` configure the three roles independently. The Analyst receives the target and optional image and may use only `search_skills` and `get_feedback`; the optimizer and optional curator remain text roles.

Role instructions are in `prompts/agents/analyst-system.md`, `optimizer-system.md`, and `curator-system.md`. Typed serde contracts are authoritative for role output.
