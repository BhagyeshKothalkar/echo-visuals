# Agentic Prompt Improver

Rust MVP for improving an image-generation prompt with an agentic LLM, Qdrant skill memory, and Redis feedback memory.

The system is deliberately tool-driven: the model is not given retrieved skills or feedback up front. It receives the target prompt and a system prompt explaining its job and tools, then decides what context it needs during generation.

## How the system works

```text
Target image-generation prompt
              |
              v
      +-------------------+
      |    Rig agent      |
      |  system prompt   |
      +-------------------+
          |       |       |
          |       |       |
          v       v       v
   search_skills  get_feedback  save_skill
          |       |       |
          v       v       v
       Qdrant    Redis     Qdrant
          |       |       |
          +--- tool results ---+
                    |
                    v
                   LLM
                    |
                    v
            final candidate prompt
                    |
                    v
       search count >= 2 and not saved?
                /          \
              yes           no
               |             |
               v             v
        deterministic      keep model
             save           result
               |
               v
          SkillRecord
               |
               v
        CandidatePrompt
```

Each iteration starts with only the target image-generation prompt. `PromptAssets` loads the editable system prompt and user template from Markdown files and renders the target into the user message.

The Rig agent then controls retrieval. `search_skills` queries Qdrant for reusable prompt-writing skills. `get_feedback` retrieves positive or negative examples previously recorded in Redis. Their results are returned to the model as tool observations, so the model can decide whether to retrieve again, change its approach, or finish.

`save_skill` lets the model explicitly persist a useful reusable skill in Qdrant. When that happens, the resulting `SkillRecord` is captured and propagated out of the generation step.

There is also a deterministic learning rule. If the model performs at least two `search_skills` calls during an iteration and did not explicitly save a skill, the harness saves the final model output as a skill. The returned `SkillRecord` becomes the candidate used by the rest of the application. With fewer than two searches and no explicit save, the model output itself is the candidate and its ID is derived deterministically from the text.

This makes Qdrant the agent's skill memory, Redis the feedback memory, and the LLM the reasoning and control layer. Retrieval is therefore dynamic rather than a separate RAG preprocessing stage.

## Data lifecycle

Skills are stored in Qdrant as reusable prompt-writing knowledge. They can be discovered with `search_skills` and persisted with `save_skill`.

Feedback is stored in Redis as candidate examples associated with `Positive` or `Negative` grades. The model can retrieve examples with `get_feedback`. User feedback is recorded explicitly by the application after a candidate has been returned.

The main application boundary is `GenerationResult`: it contains the final model text, tool-call statistics, and any `SkillRecord` created during the iteration. The harness uses this result to apply the deterministic save rule and produce the final `CandidatePrompt`.

## Run

1. Start services: `docker compose up -d`.
2. Copy `config.example.toml` to `config.toml` and edit public settings.
3. Copy `.env.example` to `.env` and set `LLM_API_KEY`.
4. Seed Qdrant: `cargo run -- init` (safe to repeat).
5. Generate once: `cargo run -- run --target "Write a clear project plan"`.
6. Iterate with feedback: `cargo run -- interactive --target "Write a clear project plan"`.

The configuration file controls service URLs, model settings, logging, and prompt paths without recompiling. Credentials are environment-only. Prompt behavior is editable in `prompts/agents/` and `prompts/templates/` Markdown files.

Tests use in-memory logic and do not require Docker or an API key: `cargo test`.
