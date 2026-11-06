# DevPilot — AI Developer Assistant

DevPilot connects to a single GitHub repository, understands its source code via RAG,
and answers developer questions using Google Gemini.

## Status

Milestone 1 complete: project skeleton, configuration management, and a basic FastAPI
app with a health check.

Milestone 2 complete: GitHub repository ingestion — validates a repository URL, fetches
its default branch and recursive tree, filters files by type/size/path, and retrieves
eligible file contents via the GitHub REST API (public repos work without a token;
private repos work with `GITHUB_TOKEN`). Retrieved contents are kept in an in-memory
store for the next milestone — the API response only returns counts/metadata.

Milestone 3 complete: RAG pipeline — chunks the currently ingested repository's files,
embeds chunks with Gemini (`gemini-embedding-001`), upserts them into Pinecone with
deterministic IDs (re-indexing overwrites instead of duplicating), and exposes semantic
search scoped to the currently ingested repository.

Milestone 4 complete: a LangGraph RAG agent answered questions about the currently
ingested repository with a fixed retrieve-then-generate pipeline. Superseded by
Milestone 5, below.

Milestone 5 complete: upgraded to a tool-calling agent (`START -> agent <-> tools -> END`,
ReAct-style). Gemini decides whether/which read-only tools to call
(`search_codebase`, `list_repository_files`, `read_repository_file`), can chain multiple
calls, and only answers once it has enough evidence — all via `POST /api/v1/chat`.

Milestone 6 complete: bug investigation — given a free-text bug report, a dedicated
LangGraph workflow systematically investigates the currently ingested repository and
returns a structured, evidence-based report via `POST /api/v1/investigate`. See
"Bug investigation" below.

Milestone 7 complete: persistent conversation memory backed by MongoDB Atlas — both
chat and investigation conversations can be persisted and resumed across requests (and
real app restarts, since MongoDB — not any in-process object — is what survives), plus
a long-term, cross-conversation memory of confirmed repository findings. See "Conversation
persistence & memory" below. Entirely optional: without `MONGODB_URI` set, every other
milestone's behavior is byte-for-byte unchanged.

Milestone 8 complete: human-in-the-loop code-change proposals — DevPilot investigates a
bug, proposes a minimal fix as a unified diff, and **pauses for real human approval**
using LangGraph's actual interrupt/resume mechanism before anything is "applied." See
"Human-in-the-loop code-change proposals" below.

Multi-agent workflows, GitHub write access, and a frontend are not implemented yet.
Code changes are proposed and can be "applied" as a reviewable artifact, but are never
written to GitHub, never committed, and never pushed — see the limitations in that
section.

## Stack

Python · LangGraph · LangChain · Google Gemini · FastAPI · Pinecone · MongoDB Atlas · Pydantic · GitHub REST API

## Project structure

```
app/
  main.py                         FastAPI app entry point; lifespan connects/closes MongoDB
  core/
    config.py                     Pydantic Settings — loads all config from environment variables
    mongodb.py                    MongoDBManager: single shared pymongo client, connect/close
    persistence.py                Factories: get_checkpointer / get_conversation_store / get_memory_service
    session.py                    Dev session identity: X-Session-Id header (see "Thread IDs" below)
  api/
    routes/
      health.py                  GET /health
      repositories.py            POST /api/v1/repositories/{ingest,index,search}
      chat.py                    POST /api/v1/chat (now with optional thread_id persistence)
      investigate.py             POST /api/v1/investigate (now with optional thread_id persistence)
      conversations.py           POST/GET /api/v1/conversations, GET/DELETE /api/v1/conversations/{thread_id}
      proposals.py               POST /api/v1/proposals, GET/.../approve/.../reject/.../diff
  agents/
    state.py                     AgentState: messages (Annotated with add_messages) + repository_id
    investigation_state.py       InvestigationState: bug report/plan/evidence/report fields + messages
    proposal_state.py            ProposalState: investigation report/draft/validation/decision fields
    message_utils.py             Shared helpers: final_answer, collect_search_sources, collect_tool_calls
    model.py                     Cached ChatGoogleGenerativeAI factory (reused across requests)
    graph.py                     Chat agent graph: START -> agent <-> tools -> END (optional checkpointer)
    investigation_graph.py       Investigation workflow graph (see "Bug investigation" below)
    proposal_graph.py            HITL proposal graph (see "Human-in-the-loop code-change proposals" below)
    nodes/
      agent_node.py              Node: invokes Gemini (tools bound) on the running conversation;
                                  reused by the chat agent, the investigator, and (indirectly) proposals
      understand_bug.py          Node: interprets the bug report (expected/actual behavior, keywords)
      plan_investigation.py      Node: turns that understanding into a focused investigation plan
      investigate_code.py        Node: the ReAct investigator (agent_node + investigator prompt)
      verify_evidence.py         Node: checks claims against real retrieved evidence
      continue_investigation.py  Node: feeds missing-evidence guidance back into another round
      analyze_root_cause.py      Node: root-cause analysis, confirmed findings vs. hypotheses
      generate_report.py         Node: pure assembly of the final report (no LLM call)
      investigate_bug.py         Node: reuses the whole investigation graph as a nested run
      generate_proposal.py       Node: drafts a minimal fix from confirmed findings (structured output)
      validate_proposal.py       Node: independent, deterministic, server-side patch validation
      request_human_approval.py  Node: the actual LangGraph interrupt() — pauses for a human decision
      apply_patch.py             Node: re-checks staleness, marks APPLIED (no filesystem writes)
    tools/
      search_codebase.py         Tool: semantic search via the existing RepositorySearchService
      list_repository_files.py   Tool: lists eligible file paths, optional prefix filter
      read_repository_file.py    Tool: reads one file's content by exact path (no traversal)
  ingestion/
    url_parser.py                Validates/parses GitHub repository URLs
    file_filter.py                Decides which files are worth retrieving (type/path/size rules)
    repository_service.py        Orchestrates: validate -> fetch tree -> filter -> fetch contents
    store.py                     In-memory holder for the current repo's retrieved files
  retrieval/
    chunking.py                  Splits file content into overlapping, line-aligned chunks
    indexing_service.py          Orchestrates: store -> chunk -> embed -> Pinecone upsert
    search_service.py            Orchestrates: embed query -> Pinecone search -> results
  services/
    github_client.py             Async GitHub REST API client (read-only)
    embedding_service.py         Async Gemini embedding client (gemini-embedding-001)
    pinecone_service.py          Async Pinecone client (index lifecycle, upsert, query)
    conversation_store.py        Conversation metadata CRUD (the `conversations` collection)
    memory_store.py              Long-term memory (LongTermMemoryService over MongoDBStore)
    patch_service.py             Diff generation + independent server-side patch validation
    proposal_store.py            Proposal + append-only audit-trail persistence
  schemas/
    repository.py                Request/response models for repository ingestion
    indexing.py                  Response model for repository indexing
    search.py                    Request/response models for semantic search
    chat.py                      Request/response models for the chat endpoint
    investigation.py             Request/response models for bug investigation
    conversation.py              Request/response models for conversation management
    proposal.py                  Request/response models for code-change proposals
tests/
  conftest.py                    Shared fixtures incl. fake Mongo collection/store, InMemorySaver use
  test_investigation_workflow.py Offline tests for the investigation graph (mocked Gemini/Pinecone)
  test_milestone5_regression.py  Confirms the chat/tool-calling agent still works after Milestone 6
  test_api_contract.py           Confirms /api/v1/chat and /api/v1/investigate response contracts
  test_mongodb.py                MongoDB config + connection manager (mocked pymongo.MongoClient)
  test_conversation_store.py     Conversation metadata CRUD (fake in-memory collection)
  test_memory_store.py           Long-term memory storage/retrieval/dedup/secret-exclusion/isolation
  test_chat_persistence_graph.py Chat graph + real LangGraph checkpointer (InMemorySaver) mechanics
  test_conversations_api.py      /api/v1/conversations endpoints end-to-end
  test_chat_api_persistence.py   /api/v1/chat with thread_id end-to-end
  test_investigate_api_persistence.py  /api/v1/investigate with thread_id end-to-end
  test_patch_service.py          Diff generation, path traversal/file-creation/staleness rejection
  test_proposal_graph.py         The real interrupt/Command(resume=...) mechanism, incl. restart simulation
  test_proposals_api.py          /api/v1/proposals endpoints end-to-end incl. audit history
```

## Bug investigation

`POST /api/v1/investigate` takes a free-text bug report and runs a dedicated LangGraph
workflow that systematically investigates the currently ingested repository, then
returns a structured, evidence-based report. It does not assume the bug exists — it
can honestly report "inconclusive" if the evidence doesn't support a conclusion.

### How it works

```
START -> understand_bug -> plan_investigation -> investigate_agent
                                                      |        ^
                                        (tool calls)  v        | (loop back with results)
                                               investigate_tools
                                                      |
                                        (no more tool calls)
                                                      v
                                              verify_evidence
                                                      |
                     +--------------------------------+-------------------------------+
                     | sufficient                      | insufficient, rounds remain  | insufficient,
                     v                                  v                              | rounds exhausted
           analyze_root_cause                continue_investigation                    v
                     |                                  |                      generate_report -> END
                     v                                  v
             generate_report -> END           investigate_agent (another round)
```

1. **understand_bug** — Gemini extracts expected vs. actual behavior, keywords, and
   candidate areas from the raw report. No code has been inspected yet.
2. **plan_investigation** — Gemini proposes a short, focused plan (which areas to look
   at and why) before any searching starts.
3. **investigate_agent / investigate_tools** — the same ReAct tool-calling loop the
   chat endpoint uses (`search_codebase`, `list_repository_files`, `read_repository_file`
   — all read-only), with an investigator-focused system prompt. Loops until Gemini
   stops requesting tools and gives its own summary.
4. **verify_evidence** — re-derives evidence directly from the repository store and the
   actual tool-call arguments (never from the investigator's prose), then asks Gemini to
   check the investigator's summary against that real evidence: is each claim backed by
   an actual snippet, not just a plausible-sounding file name? If not sufficient and
   rounds remain, **continue_investigation** feeds back what's missing and loops to
   investigate_agent again; if rounds are exhausted, the workflow goes straight to
   generate_report as **inconclusive** rather than guessing.
5. **analyze_root_cause** — only reached once evidence is judged sufficient. Explains
   the root cause from the verified findings, explicitly separating `confirmed_findings`
   (directly backed by evidence) from `hypotheses` (plausible but unconfirmed).
6. **generate_report** — makes no LLM call at all; it's pure assembly of everything the
   earlier (already evidence-grounded) nodes produced, so the report itself can't
   introduce new fabrication.

The outer verify → continue → investigate loop is capped by `INVESTIGATION_MAX_ROUNDS`
(default 3) so an unresolvable investigation stops with an honest inconclusive report
instead of looping forever; `INVESTIGATION_RECURSION_LIMIT` (default 60) is a graph-level
safety net on top of that.

### Request

```bash
curl -X POST http://127.0.0.1:8000/api/v1/investigate \
  -H "Content-Type: application/json" \
  -d '{"bug_report": "Deactivated staff members can still access the application after their accounts are disabled."}'
```

### Response structure

```json
{
  "status": "complete",
  "bug_summary": "...",
  "expected_behavior": "...",
  "actual_behavior": "...",
  "investigation_performed": ["search_codebase(query='...') — ok", "read_repository_file(file_path='...') — ok"],
  "root_cause": "...",
  "confirmed_findings": ["..."],
  "hypotheses": ["..."],
  "supporting_evidence": [
    {"source_tool": "read_repository_file", "file_path": "...", "start_line": 1, "end_line": 20, "snippet": "..."}
  ],
  "affected_files": ["..."],
  "confidence_level": "high",
  "recommended_fix_direction": "...",
  "remaining_unknowns": ["..."],
  "tool_calls": [{"tool": "search_codebase", "arguments": {"query": "..."}, "status": "success"}]
}
```

`status` is `"complete"` or `"inconclusive"`. When inconclusive, `confidence_level` is
also `"inconclusive"`, `root_cause` says plainly that evidence was insufficient, and
`remaining_unknowns` lists what's missing — the endpoint never invents a root cause to
force a confident-sounding answer.

`confidence_level` (`high`/`medium`/`low`/`inconclusive`) and the `confirmed_findings`
vs. `hypotheses` split are how confidence is represented: `confirmed_findings` are only
ever things `verify_evidence` already checked against real retrieved content;
`hypotheses` are explicitly flagged as not directly confirmed. `supporting_evidence`
snippets are never LLM-generated text — they're sliced directly from the in-memory
repository store (for `search_codebase` hits) or the tool's actual recorded arguments
(for `read_repository_file` calls), so every snippet is traceable to real ingested
content.

## Conversation persistence & memory

MongoDB Atlas backs two independent capabilities, both optional:

1. **Short-term memory** — a conversation's actual message history (chat Q&A turns,
   tool calls, tool results), persisted per `thread_id` via LangGraph's checkpointer.
2. **Long-term memory** — a small set of durable, cross-conversation facts about a
   repository (currently: confirmed bug-investigation findings), scoped to
   `(session, repository)`, retrieved and injected into future chat conversations.

**MongoDB and Pinecone have different jobs and neither replaces the other.** Pinecone
remains solely responsible for repository *code* embeddings and semantic code search
(Milestone 3) — that index is untouched by this milestone. MongoDB is responsible for
*conversation* state and memory — it never stores code embeddings, and the Pinecone
code index is never migrated or duplicated into MongoDB.

Everything in this section is additive and optional: leave `MONGODB_URI` unset and
`/chat`, `/investigate`, and all earlier endpoints behave exactly as in Milestones 1–6,
with zero MongoDB reads or writes.

### Setup

1. Create a free Atlas cluster at https://cloud.mongodb.com, add a database user, and
   allow your IP (or `0.0.0.0/0` for local development only).
2. Copy the connection string into `.env` as `MONGODB_URI` — never paste it into chat;
   edit `.env` directly, and never commit it (`.env` is git-ignored).
3. `MONGODB_DATABASE` defaults to `devpilot` if unset.

```bash
# .env
MONGODB_URI=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/
MONGODB_DATABASE=devpilot
```

### Database and collections (all in the `MONGODB_DATABASE` database)

| Collection          | Written by                              | Contents |
|---------------------|------------------------------------------|----------|
| `conversations`     | `app/services/conversation_store.py`      | Conversation identity/ownership: `thread_id` (`_id`), `session_id`, `repository_id`, `workflow`, `title`, `metadata`, `status`, `created_at`, `updated_at`. |
| `checkpoints`, `checkpoint_writes` | `langgraph-checkpoint-mongodb` (`MongoDBSaver`) | Full LangGraph state per thread — messages, tool calls/results, in-progress investigation state. Never read/written directly by DevPilot code; only through the checkpointer API. |
| `memories`           | `langgraph-store-mongodb` (`MongoDBStore`) via `app/services/memory_store.py` | Long-term memories: `{content, kind, source}`, keyed by a content hash under namespace `(session_id, repository_id)`. |

### Short-term memory: the LangGraph checkpointer

`MongoDBSaver` (from `langgraph-checkpoint-mongodb`) wraps a single shared, synchronous
`pymongo.MongoClient` — its async methods (`aget_tuple`, `aput`, …, used automatically
by `graph.ainvoke()`) internally run that sync client's calls in a thread-pool executor,
which is the supported way to use a sync-only driver from async code without blocking
the event loop. `build_agent_graph(..., checkpointer=...)` and
`build_investigation_graph(..., checkpointer=...)` both take it as an optional
parameter — omitted (the default), behavior is identical to Milestones 5/6.

When a request includes a `thread_id`, LangGraph:
1. loads that thread's last checkpoint (restoring its full `messages` history),
2. merges in the new input — for `messages` specifically, the `add_messages` reducer
   *appends* rather than overwrites, so the new turn is added to history, not duplicated
   or lost,
3. saves a new checkpoint after the run.

This is why "persists across an application restart" holds without any special
handling: every request builds a brand-new graph object (see `app/api/routes/chat.py`)
— there's never a long-lived graph instance to lose. The only thing that must survive a
restart is MongoDB itself, which is a separate process by design.

**Chat vs. investigation threads behave differently on reuse**, and this is a
deliberate design decision:
- A **chat** thread resumes normally — that's the entire point of short-term memory
  for chat.
- An **investigation** thread never resumes mid-workflow. Milestone 6's investigation
  graph runs start-to-finish in one call; there is no valid "paused halfway" state to
  continue from. Reusing an investigation's `thread_id` means "run a fresh
  investigation, keep grouping it under this conversation record" — the route clears
  any prior checkpoint for that thread (`checkpointer.adelete_thread`) before invoking,
  so a new bug report can never inherit stale evidence, verified findings, or a stale
  root-cause analysis from a previous run on the same thread.

### Long-term memory: cross-conversation facts

`MongoDBStore` (from `langgraph-store-mongodb`) is used in plain **filter/namespace
mode**, not its semantic `query` mode. The semantic mode needs an Atlas Vector Search
index (auto-created on first use) plus a second embedding pipeline alongside the one
already used for code search — real operational complexity this milestone's actual
requirement ("retrieve relevant memories") doesn't need, since the memory volume per
`(session, repository)` is small and deliberately curated rather than ranked.

**What gets stored, and when:** only `confirmed_findings` from a *completed*
investigation (never `hypotheses` — unverified LLM assumptions are never promoted to
memory) are written automatically, with zero extra LLM calls — this reuses output the
investigation graph already produced and verified in `verify_evidence`/
`analyze_root_cause`. Plain chat does not write memories in this milestone (to avoid
"store every message" sprawl); it only *reads* them.

**What gets retrieved, and how:** every `/chat` call (ephemeral or persisted) recalls
all memories for `(session_id, repository_id)` and injects them as extra system-prompt
context — explicitly labeled as prior, unverified-in-this-conversation context the
agent should re-check with its tools before relying on, not ground truth.

**Deduplication:** each memory's key is a hash of `(kind, content)`. Storing the exact
same finding again refreshes it instead of creating a duplicate; storing a *reworded*
version of the same fact creates a separate entry (a known limitation — see below).

**Isolation:** the namespace *is* `(session_id, repository_id)` — a lookup scoped to
one namespace structurally cannot return another session's or another repository's
memories. There is no cross-namespace query path in `LongTermMemoryService` at all.

**Safety:** `remember()` refuses to store content matching secret-*shaped* values
(`key=value` assignments, embedded connection-string credentials, known API-key
prefixes, bearer tokens) — but not bare words like "token"/"password"/"secret" alone,
since legitimate auth-related findings routinely (and safely) name those concepts.

### Thread IDs, sessions, and ownership

Three identifiers are never conflated:
- **`thread_id`** identifies one conversation (chat or investigation).
- **session identity** identifies who owns it.
- **`repository_id`** identifies which repository it's scoped to.

DevPilot has no real authentication yet. Rather than pretending anonymous requests are
authenticated users, callers identify themselves via an `X-Session-Id` header
(`app/core/session.py`) — an explicit, documented placeholder. If omitted, the server
mints a new one and returns it in the response headers; reuse it to keep owning your
conversations. **This is not authentication** — anyone who learns a session ID can act
as it. Replace this with a real identity source once auth exists.

Every conversation and chat/investigate call with a `thread_id` checks that the
conversation's stored `session_id` matches the caller's. A mismatch (or a
nonexistent `thread_id`) returns **404**, not 403 — this avoids confirming to a
guesser that a given `thread_id` exists at all. A `thread_id` also can't be used across
workflows (a chat `thread_id` rejected by `/investigate` and vice versa, `409`) or
across repositories (continuing a thread scoped to a repository other than the one
currently ingested, `409` — DevPilot's tools only ever operate on "the currently
ingested repository," so silently answering under the wrong one would be a real
correctness bug, not just a UX nicety).

### New endpoints

```
POST   /api/v1/conversations                Create a conversation (chat or investigation)
GET    /api/v1/conversations                List the caller's conversations (paginated)
GET    /api/v1/conversations/{thread_id}    Conversation detail, with restored messages
DELETE /api/v1/conversations/{thread_id}    Delete a conversation
```

None of these ever return raw LangGraph checkpoint documents — `GET /{thread_id}` maps
the checkpoint's messages into a plain `{role, content, tool_name}` shape first.

**Create a conversation:**
```bash
curl -X POST http://127.0.0.1:8000/api/v1/conversations \
  -H "Content-Type: application/json" \
  -H "X-Session-Id: my-dev-session" \
  -d '{"workflow": "chat", "title": "Investigating the auth flow"}'
# -> {"thread_id": "a1b2c3...", "repository_id": "acme/webapp", "workflow": "chat",
#     "title": "...", "status": "active", "created_at": "...", "updated_at": "..."}
```

**Resume it in chat** (pass the returned `thread_id`):
```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" -H "X-Session-Id: my-dev-session" \
  -d '{"question": "How does login work?", "thread_id": "a1b2c3..."}'
# later, same thread_id -> the agent sees the first question + answer as prior context
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" -H "X-Session-Id: my-dev-session" \
  -d '{"question": "And where are sessions invalidated?", "thread_id": "a1b2c3..."}'
```

`ChatRequest`/`InvestigateRequest` gained an additive, optional `thread_id`; omitting it
reproduces exact Milestone 5/6 behavior. `ChatResponse`/`InvestigationResponse` gained an
additive `thread_id` field that echoes back what was used (`null` if ephemeral).

**Deletion policy:** `DELETE /api/v1/conversations/{thread_id}` removes the
conversation's checkpoint (`checkpointer.adelete_thread`) and its metadata document.
**It does not delete long-term memories that conversation produced.** This is a
deliberate policy, not an oversight: a confirmed finding about the codebase remains
true and useful after the conversation that surfaced it is gone — each memory keeps its
originating `thread_id` in `source` for audit purposes regardless.

### Known limitations

- Memory deduplication is exact-content-hash based — a reworded version of an
  already-stored finding creates a second entry rather than merging with it.
- Long-term memory retrieval is namespace-filter based, not semantically ranked;
  fine for the small, curated volume this milestone produces, but would need the
  (deliberately deferred) Atlas Vector Search path to scale to many memories.
- Chat history accumulates indefinitely within a thread — there is no summarization/
  trimming of long conversations yet, so a very long-running thread will eventually
  grow its context window usage unbounded.
- Plain chat conversations don't themselves write long-term memories (only completed
  investigations do) — a deliberate scope boundary for this milestone, not a bug.

## Human-in-the-loop code-change proposals

`POST /api/v1/proposals` takes a bug report and runs investigation through to a
proposed fix — but the fix is never applied without a real, separate human decision.
The developer stays in control of every code modification: a proposal pauses at an
actual LangGraph interrupt and stays paused in MongoDB until `/approve` or `/reject` is
called explicitly. Unlike chat/investigate, this workflow has **no ephemeral mode** —
pausing for approval only means something if the pause can actually be persisted, so
MongoDB is required here.

### Architecture

```
START -> investigate_bug
             |
  (status != FAILED)              (status == FAILED, e.g. investigation was inconclusive)
             v                                  v
     generate_proposal                         END
             |
      validate_proposal
             |
  (status != FAILED)              (status == FAILED, e.g. path/size/extension violation)
             v                                  v
  request_human_approval                       END
   [INTERRUPT -- pauses here;
    resumes via Command(resume=...)]
             |
  (decision == "approved")        (decision == "rejected" -- status already REJECTED)
             v                                  v
       apply_patch                             END
             v
            END
```

1. **investigate_bug** — reuses Milestone 6's entire investigation graph as a nested
   run (not duplicated logic — a direct call to `build_investigation_graph(...).ainvoke(...)`).
   If the investigation doesn't reach a *confirmed* root cause (status `"complete"`),
   the proposal graph stops immediately with `status: FAILED` — **a fix is never drafted
   from a hypothesis alone**.
2. **generate_proposal** — asks Gemini for the smallest reasonable fix, grounded only in
   `confirmed_findings` (hypotheses are passed along as explicitly unverified context,
   never as something to fix). Gathers the real original content of the investigation's
   `affected_files` directly from the repository store first; if Gemini says it needs to
   see another file, one bounded extra context pass happens before the final draft — not
   an open-ended tool-calling loop. Output is structured (`ProposalDraft`): per-file
   `proposed_content` (the complete new file, not a snippet), an explanation, risk level,
   and explicit assumptions. **No diff is ever requested from or trusted from the LLM.**
3. **validate_proposal** — no LLM call. Independent, deterministic, server-side
   validation (`app/services/patch_service.py`): every unified diff is computed here from
   `(original, proposed)` content; every file path is checked against the exact,
   known set of files in the currently ingested repository (structurally identical to how
   `read_repository_file` prevents traversal — there's no filesystem path at all, only
   exact-match lookups, so `"../../etc/passwd"` or an absolute path simply matches
   nothing); unsupported file types, file creation, duplicate files, and oversized diffs
   are all rejected. A proposal that fails here never reaches a human at all.
4. **request_human_approval** — the real `langgraph.types.interrupt()`, not a simulated
   boolean. The graph genuinely pauses — `graph.ainvoke(...)` returns with a
   `"__interrupt__"` key instead of completing, and the paused state is persisted via the
   `MongoDBSaver` checkpointer. There is no code path from here to `apply_patch` without
   this pause actually happening: it's the gate itself, not a check guarding it.
5. **apply_patch** — only reachable from the "approved" branch; the graph's own structure
   (not a flag) guarantees a rejected or still-pending proposal can never reach it. Before
   marking `APPLIED`, it re-checks every file's content hash against what's currently
   ingested — if the repository changed since the proposal was drafted, it fails as stale
   rather than applying a patch against content that no longer matches.

### LangGraph interrupt/resume, concretely

```python
# inside request_human_approval_node — paraphrased
decision_payload = interrupt({"proposal_id": ..., "risk_level": ..., "affected_files": [...]})
# ^ on first run: raises internally, LangGraph persists state via the checkpointer,
#   and graph.ainvoke(...) returns normally with a "__interrupt__" key in its result.
# ^ on resume (Command(resume={"decision": "approved", "reason": "..."})):
#   interrupt(...) returns that dict immediately, and the rest of the node runs.
decision = decision_payload["decision"]
```

Resuming:
```python
await graph.ainvoke(
    Command(resume={"decision": "approved", "reason": "Looks correct."}),
    config={"configurable": {"thread_id": proposal_id}},
)
```

Everything in the node *before* the `interrupt()` call re-runs on resume too — this node
has nothing before it but the interrupt call itself, which is safe by construction (see
the module docstring in `app/agents/nodes/request_human_approval.py`).

### MongoDB's role here

The same `MongoDBSaver` checkpointer from Milestone 7, nothing new: pausing is just a
checkpoint like any other, and resuming is just loading it back and continuing. The
`proposals` collection (API-facing document: diff, validation results, status, decision)
and `proposal_audit` collection (append-only event log) are separate from the checkpoint
internals — same decoupling as the `conversations` collection vs. chat/investigation
checkpoints in Milestone 7, so API responses never expose raw checkpoint documents.

### API endpoints

```
POST   /api/v1/proposals                    Investigate + draft a proposal; pauses at approval
GET    /api/v1/proposals                    List the caller's proposals (paginated)
GET    /api/v1/proposals/{proposal_id}      Retrieve a proposal
GET    /api/v1/proposals/{proposal_id}/diff Retrieve just the unified diff
POST   /api/v1/proposals/{proposal_id}/approve   Resume the paused graph with "approved"
POST   /api/v1/proposals/{proposal_id}/reject    Resume the paused graph with "rejected"
```

Every proposal gets its own freshly-generated `thread_id` (equal to `proposal_id` by
design — there's no reuse semantic here the way chat/investigate threads have, since a
proposal's lifecycle only ever happens once). Ownership (404, not 403, on mismatch — see
Milestone 7) and repository-consistency (409 if the currently ingested repository has
changed since the proposal was created) are both checked before any decision is acted on.

**A decision can only be made once.** `/approve` and `/reject` both check the proposal is
still `PENDING_APPROVAL` *before ever touching the graph* — a proposal that's already
`APPROVED`/`REJECTED`/`APPLIED`/`FAILED` gets a clean `409`, never a re-run. This is what
prevents replaying or duplicating a decision, independent of the graph's own structural
guarantee that a rejected branch can never reach `apply_patch`.

### Example

```bash
curl -X POST http://127.0.0.1:8000/api/v1/proposals \
  -H "Content-Type: application/json" -H "X-Session-Id: dev-1" \
  -d '{"bug_report": "Deactivated staff members can still access the application after their account is disabled."}'
```
```json
{
  "proposal_id": "41e5...c1", "thread_id": "41e5...c1", "repository_id": "octocat/Spoon-Knife",
  "root_cause": "The layout uses rigid, fixed-width CSS properties without responsive constraints.",
  "explanation": "...", "risk_level": "low",
  "proposed_changes": [{
    "file_path": "styles.css", "original_content_hash": "027fb1...",
    "unified_diff": "--- a/styles.css\n+++ b/styles.css\n@@ ...\n-  width:384px;\n+  max-width: 384px;\n+  width: 100%;\n ...",
    "change_summary": "Replaced fixed widths with max-width to allow responsive scaling."
  }],
  "validation_results": {"valid": true, "issues": []},
  "status": "PENDING_APPROVAL", "decision": null
}
```
```bash
curl -X POST http://127.0.0.1:8000/api/v1/proposals/41e5...c1/approve \
  -H "Content-Type: application/json" -H "X-Session-Id: dev-1" \
  -d '{"reason": "Looks correct, approving."}'
# -> same body, "status": "APPLIED", "decision": "approved"

curl -X POST http://127.0.0.1:8000/api/v1/proposals/41e5...c1/approve \
  -H "X-Session-Id: dev-1" -d '{}'
# -> 409: already decided, not re-applied
```

### Safe application — what "applying" actually means

DevPilot ingests repositories through the GitHub REST API into an in-memory
`RepositoryStore` (Milestone 2) — **there is no writable Git working directory**, and
this milestone does not invent one. "Applying" a patch here means:
1. Re-verifying the proposal isn't stale (the file's content hash still matches what's
   currently ingested).
2. Marking the proposal `APPLIED`.

The "applied" result **is** the proposal's own `proposed_content`/`unified_diff` —
retrievable via `GET /api/v1/proposals/{id}` and `.../diff` — an approved-patch artifact,
not a mutation of anything. **The original ingested repository snapshot is never
written to.** There is no filesystem write, no shell execution, no running of repository
code, and — critically — **no GitHub write access, no commit, and no push, at any
point.** Getting an approved change into the real repository remains a manual step for
the developer (e.g. applying the diff locally and opening a PR themselves).

### Audit history

Every proposal creation, validation outcome, decision, and apply result is recorded as
an append-only event in `proposal_audit` (`created`, `pending_approval` /
`validation_failed`, `approved` / `rejected`, `applied` / `apply_failed`) — approving or
rejecting a proposal updates the `proposals` document's status but never overwrites or
removes a prior audit entry.

### Known limitations

- No real Git working directory — "applying" is an artifact (diff + full new file
  content), not a file-system or repository mutation. See "Safe application" above.
- No new-file creation or deletion support — only modifications to files already present
  in the ingested repository are supported, by design (`validate_proposal` rejects
  anything else as `unknown_file`).
- `generate_proposal`'s "read additional files" capability is one bounded extra context
  pass, not an open-ended ReAct tool-calling loop — sufficient for small, scoped fixes,
  but not for a fix that turns out to need iterative exploration across many files.
- Proposals have no expiry/TTL yet — `ProposalStatus.EXPIRED` exists in the schema for
  future use but nothing currently transitions a proposal into it.
- After a real process restart, the in-memory `RepositoryStore` (Milestone 2) is empty —
  approving a pending proposal requires re-ingesting the same repository first, even
  though the proposal itself survives the restart intact in MongoDB. This was confirmed
  directly: approval correctly fails closed (409, not a wrong-repository apply) until the
  repository is re-ingested.

## Setup

```bash
# From this directory, with venv already created at ./venv
./venv/Scripts/python.exe -m pip install -r requirements.txt

# Copy and fill in secrets (never commit .env)
cp .env.example .env
```

## Testing

```bash
./venv/Scripts/python.exe -m pytest
```

All tests are offline and need no API keys or a real MongoDB/Atlas cluster — Gemini is
replaced with a scripted fake model, Pinecone search with a substring-matching fake,
and MongoDB with two kinds of fakes:
- `FakeCollection`/`FakeLongTermStore` (`tests/conftest.py`) — minimal in-memory stand-ins
  for exactly the pymongo/Store methods `ConversationStore`/`LongTermMemoryService` call,
  used to test *our* logic without MongoDB's wire protocol at all.
- `langgraph.checkpoint.memory.InMemorySaver` for checkpointer tests — not a MongoDB
  fake, but the real `BaseCheckpointSaver` interface `MongoDBSaver` also implements, so
  these tests verify our actual usage of that interface (thread restoration, isolation).

If you have a real Atlas cluster available, you can additionally point `MONGODB_URI` at
it and exercise the endpoints manually via `curl`/the `/docs` UI — the automated suite
intentionally does not require or attempt this.

## Running

```bash
./venv/Scripts/python.exe -m uvicorn app.main:app --reload
```

Then visit:
- http://127.0.0.1:8000/ — root status
- http://127.0.0.1:8000/health — health check
- http://127.0.0.1:8000/docs — interactive API docs

## Security notes

- No API keys or tokens are hardcoded; all secrets come from environment variables via `app/core/config.py`.
- `.env` is git-ignored.
- GitHub repository access is read-only. Code fetched from repositories is never executed.
- `X-Session-Id` (`app/core/session.py`) is an explicit pre-authentication placeholder,
  not real auth — anyone who learns a session ID can act as it. Replace it with a real
  identity source before relying on it in production.
- Conversation/thread access is ownership-checked on every call; mismatches return 404
  (not 403) to avoid confirming a guessed `thread_id` exists.
- Long-term memory writes are filtered against secret-shaped content (API keys,
  embedded connection-string credentials, bearer tokens) before being stored —
  see "Long-term memory" above. `.env` values are never read into memory content in the
  first place; this filter is a defense-in-depth safety net, not the primary control.
