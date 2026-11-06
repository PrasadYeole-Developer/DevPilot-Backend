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

Multi-agent workflows, conversation memory, code modification, GitHub write access, and
a frontend are not implemented yet.

## Stack

Python · LangGraph · LangChain · Google Gemini · FastAPI · Pinecone · Pydantic · GitHub REST API

## Project structure

```
app/
  main.py                         FastAPI app entry point
  core/
    config.py                     Pydantic Settings — loads all config from environment variables
  api/
    routes/
      health.py                  GET /health
      repositories.py            POST /api/v1/repositories/{ingest,index,search}
      chat.py                    POST /api/v1/chat
      investigate.py             POST /api/v1/investigate
  agents/
    state.py                     AgentState: messages (Annotated with add_messages) + repository_id
    investigation_state.py       InvestigationState: bug report/plan/evidence/report fields + messages
    message_utils.py             Shared helpers: final_answer, collect_search_sources, collect_tool_calls
    model.py                     Cached ChatGoogleGenerativeAI factory (reused across requests)
    graph.py                     Chat agent graph: START -> agent <-> tools -> END
    investigation_graph.py       Investigation workflow graph (see "Bug investigation" below)
    nodes/
      agent_node.py              Node: invokes Gemini (tools bound) on the running conversation;
                                  reused by both the chat agent and the investigator
      understand_bug.py          Node: interprets the bug report (expected/actual behavior, keywords)
      plan_investigation.py      Node: turns that understanding into a focused investigation plan
      investigate_code.py        Node: the ReAct investigator (agent_node + investigator prompt)
      verify_evidence.py         Node: checks claims against real retrieved evidence
      continue_investigation.py  Node: feeds missing-evidence guidance back into another round
      analyze_root_cause.py      Node: root-cause analysis, confirmed findings vs. hypotheses
      generate_report.py         Node: pure assembly of the final report (no LLM call)
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
  schemas/
    repository.py                Request/response models for repository ingestion
    indexing.py                  Response model for repository indexing
    search.py                    Request/response models for semantic search
    chat.py                      Request/response models for the chat endpoint
    investigation.py             Request/response models for bug investigation
tests/
  conftest.py                    Shared fixtures: sample repo store, fake search service, scripted chat model
  test_investigation_workflow.py Offline tests for the investigation graph (mocked Gemini/Pinecone)
  test_milestone5_regression.py  Confirms the chat/tool-calling agent still works after Milestone 6
  test_api_contract.py           Confirms /api/v1/chat and /api/v1/investigate response contracts
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

All tests are offline — Gemini is replaced with a scripted fake model and Pinecone
search is replaced with a substring-matching fake, so the suite needs no API keys and
makes no network calls.

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
