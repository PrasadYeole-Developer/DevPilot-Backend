"""
LangGraph node: generate_proposal.

Drafts the smallest reasonable fix from a VERIFIED investigation's confirmed
findings only (the graph only reaches this node when investigate_bug's
report.status == "complete" — see app.agents.proposal_graph's routing).
Hypotheses are passed along as explicitly-labeled unverified context, never
as something to fix.

Gathers file content directly from the ground-truth RepositoryStore (the
same source app.agents.tools.read_repository_file uses) rather than through
a full ReAct tool-calling loop — this node makes one (or, if the model asks
for more context, two) deterministic structured-output calls, not an
open-ended agent loop, which keeps proposal generation bounded and simple
while still letting the model see additional files if the investigation's
affected_files list wasn't sufficient.
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.proposal_state import ProposalState
from app.ingestion.store import RepositoryStore
from app.schemas.investigation import InvestigationResponse
from app.schemas.proposal import ProposalDraft

SYSTEM_PROMPT = """You propose the smallest reasonable code fix for a CONFIRMED root cause \
in a software repository.

Rules:
- Base your fix ONLY on the confirmed_findings and root_cause provided below — never on \
hypotheses alone. Hypotheses are unverified context, not something to fix.
- Propose the smallest change that addresses the root cause. Do not refactor unrelated \
code, rename things, reformat untouched lines, or "clean up" anything you weren't asked \
to change.
- Never invent functions, imports, files, classes, or APIs that are not already visible \
in the provided file contents.
- Preserve all existing code you are not specifically changing — proposed_content for \
each file must be the file's COMPLETE content with your fix applied, not a snippet or a diff.
- Only propose changes to files whose content you were given below. If you genuinely need \
to see another file first, list its exact path in additional_files_needed and leave \
changes empty — you will be given that file's content and asked again.
- List any assumption you had to make, explicitly, in assumptions.
- Set risk_level honestly: "low" for a narrow, obviously-safe change; "high" if the change \
touches logic that looks broadly used or security-sensitive."""


async def generate_proposal_node(state: ProposalState, chat_model: BaseChatModel, store: RepositoryStore) -> dict:
    report = state["investigation_report"]
    repository = store.get_current()
    if repository is None:
        return {"status": "FAILED", "error": "No repository is currently ingested."}

    files_by_path = {f.path: f for f in repository.files}
    gathered: dict[str, str] = {
        path: files_by_path[path].content for path in report.affected_files if path in files_by_path
    }

    structured_model = chat_model.with_structured_output(ProposalDraft)
    draft: ProposalDraft = await structured_model.ainvoke(_build_messages(report, gathered))

    newly_requested = [
        path for path in draft.additional_files_needed if path in files_by_path and path not in gathered
    ]
    if newly_requested and not draft.changes:
        for path in newly_requested:
            gathered[path] = files_by_path[path].content
        draft = await structured_model.ainvoke(_build_messages(report, gathered))

    return {"proposal_draft": draft, "status": "DRAFT"}


def _build_messages(report: InvestigationResponse, gathered: dict[str, str]) -> list:
    context = (
        "\n\n".join(f"--- {path} ---\n{content}" for path, content in gathered.items())
        or "(no file content gathered yet)"
    )
    prompt = (
        f"Confirmed root cause: {report.root_cause}\n\n"
        "Confirmed findings:\n" + "\n".join(f"- {f}" for f in report.confirmed_findings) + "\n\n"
        "Unverified hypotheses (context only — do not fix these alone):\n"
        + ("\n".join(f"- {h}" for h in report.hypotheses) or "(none)") + "\n\n"
        f"Affected files identified by the investigation: {', '.join(report.affected_files)}\n\n"
        f"Current file content:\n{context}"
    )
    return [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
