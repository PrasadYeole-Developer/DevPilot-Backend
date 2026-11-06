"""
Shared helpers for reading a finished agent conversation's message list.

Extracted from app.api.routes.chat (Milestone 5) so the investigation graph
(Milestone 6) can reuse the exact same tool-call/source extraction logic
instead of re-implementing it.
"""

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from app.schemas.chat import ToolCallRecord


def final_answer(messages: list[BaseMessage]) -> str | None:
    """The last AIMessage that isn't itself a tool call — i.e. the agent's final reply."""
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            return message.text
    return None


def collect_search_sources(messages: list[BaseMessage]) -> list[dict]:
    """Deduplicated file_path/start_line/end_line/similarity_score dicts from search_codebase calls."""
    seen: set[tuple[str, int, int]] = set()
    sources: list[dict] = []
    for message in messages:
        if isinstance(message, ToolMessage) and message.name == "search_codebase" and message.artifact:
            for item in message.artifact:
                key = (item["file_path"], item["start_line"], item["end_line"])
                if key not in seen:
                    seen.add(key)
                    sources.append(item)
    return sources


def collect_tool_calls(messages: list[BaseMessage]) -> list[ToolCallRecord]:
    """Every tool call made in the conversation, paired with its outcome status."""
    tool_messages_by_call_id = {
        message.tool_call_id: message for message in messages if isinstance(message, ToolMessage)
    }
    records: list[ToolCallRecord] = []
    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            for call in message.tool_calls:
                result = tool_messages_by_call_id.get(call["id"])
                records.append(
                    ToolCallRecord(
                        tool=call["name"],
                        arguments=call["args"],
                        status=getattr(result, "status", "unknown"),
                    )
                )
    return records
