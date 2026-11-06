"""
LangGraph node: verify_evidence — checks whether retrieved evidence actually
supports the investigator's claims.

Evidence is derived directly from the in-memory repository store and the
recorded tool-call arguments — never from the investigator's own free-text
summary — so every EvidenceItem is traceable to real ingested content rather
than something an LLM merely asserted.
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from app.agents.investigation_state import InvestigationState
from app.agents.message_utils import final_answer
from app.ingestion.store import RepositoryStore, RetrievedFile
from app.schemas.investigation import EvidenceAssessment, EvidenceItem

MAX_SNIPPET_CHARS = 800

SYSTEM_PROMPT = """You are a skeptical reviewer checking a bug investigation.

You will be given the evidence actually retrieved from the repository (real file
content, not a description of it) and the investigator's own summary of their
findings. Determine:
- Whether each claim in the investigator's summary is actually backed by the
  retrieved evidence — not just plausible-sounding, but literally shown in a
  snippet below.
- Any contradictions between the claims and the evidence.
- What evidence is still missing to confirm or refute the bug.
- Whether the evidence collected so far is sufficient to proceed to a root-cause
  analysis, or whether more investigation is needed.

A file path or a similarity score alone is NOT evidence of behavior — only the
actual code/content counts. If no tool was ever called, evidence is NOT sufficient."""


async def verify_evidence_node(
    state: InvestigationState, chat_model: BaseChatModel, store: RepositoryStore
) -> dict:
    evidence = extract_evidence(state["messages"], store)
    investigator_summary = final_answer(state["messages"])

    structured_model = chat_model.with_structured_output(EvidenceAssessment)
    prompt = (
        f"Bug report: {state['bug_report']}\n\n"
        f"Investigator's summary:\n"
        f"{investigator_summary or '(The investigator made no tool calls and gave no summary.)'}\n\n"
        f"Actual evidence retrieved ({len(evidence)} item(s)):\n{_format_evidence(evidence)}"
    )
    assessment: EvidenceAssessment = await structured_model.ainvoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
    )

    return {"evidence": evidence, "evidence_assessment": assessment, "status": "verifying"}


def extract_evidence(messages: list[BaseMessage], store: RepositoryStore) -> list[EvidenceItem]:
    repository = store.get_current()
    files_by_path: dict[str, RetrievedFile] = {f.path: f for f in repository.files} if repository else {}

    tool_calls_by_id: dict[str, dict] = {}
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                tool_calls_by_id[call["id"]] = call

    evidence: list[EvidenceItem] = []
    seen: set[tuple] = set()

    for message in messages:
        if not isinstance(message, ToolMessage) or message.status != "success":
            continue

        if message.name == "search_codebase" and message.artifact:
            for item in message.artifact:
                key = ("search_codebase", item["file_path"], item["start_line"], item["end_line"])
                if key in seen:
                    continue
                seen.add(key)
                snippet = _slice_file(files_by_path.get(item["file_path"]), item["start_line"], item["end_line"])
                evidence.append(
                    EvidenceItem(
                        source_tool="search_codebase",
                        file_path=item["file_path"],
                        start_line=item["start_line"],
                        end_line=item["end_line"],
                        snippet=snippet,
                    )
                )

        elif message.name == "read_repository_file":
            call = tool_calls_by_id.get(message.tool_call_id, {})
            path = call.get("args", {}).get("file_path")
            if not path:
                continue
            key = ("read_repository_file", path, None, None)
            if key in seen:
                continue
            seen.add(key)
            file = files_by_path.get(path)
            if file is None:
                continue
            evidence.append(
                EvidenceItem(
                    source_tool="read_repository_file",
                    file_path=path,
                    start_line=1,
                    end_line=file.content.count("\n") + 1,
                    snippet=_truncate(file.content, MAX_SNIPPET_CHARS),
                )
            )

    return evidence


def _slice_file(file: RetrievedFile | None, start_line: int, end_line: int) -> str:
    if file is None:
        return ""
    lines = file.content.splitlines()
    snippet = "\n".join(lines[max(0, start_line - 1) : end_line])
    return _truncate(snippet, MAX_SNIPPET_CHARS)


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "... [truncated]"


def _format_evidence(evidence: list[EvidenceItem]) -> str:
    if not evidence:
        return "(No evidence was retrieved — no tool calls succeeded.)"
    return "\n\n".join(
        f"--- {item.file_path} "
        f"({f'lines {item.start_line}-{item.end_line}' if item.start_line else 'full file'}, "
        f"via {item.source_tool}) ---\n{item.snippet}"
        for item in evidence
    )
