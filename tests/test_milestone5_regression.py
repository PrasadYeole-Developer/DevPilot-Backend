"""
Regression tests confirming Milestone 5's chat/tool-calling agent still works
after Milestone 6's changes — specifically, the `agent_node` refactor (adding
an optional `system_prompt` parameter) and extracting `_collect_sources`/
`_collect_tool_calls`/`_final_answer` out of chat.py into app.agents.message_utils.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphRecursionError

from app.agents.graph import build_agent_graph
from app.agents.message_utils import collect_search_sources, collect_tool_calls, final_answer
from app.agents.tools import build_tools
from app.schemas.chat import ChatResponse

pytestmark = pytest.mark.anyio


class _FakeChatModel:
    def __init__(self, responses):
        self._responses = iter(responses)
        self.call_count = 0

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        self.call_count += 1
        return next(self._responses)


async def test_agent_graph_tool_call_loop_still_works(sample_store, fake_search_service):
    tools = build_tools(search_service=fake_search_service, store=sample_store)
    model = _FakeChatModel(
        [
            AIMessage(content="", tool_calls=[{"name": "search_codebase", "args": {"query": "session"}, "id": "c1"}]),
            AIMessage(content="Sessions are validated by checking the token.", tool_calls=[]),
        ]
    )

    graph = build_agent_graph(chat_model=model, tools=tools)
    final_state = await graph.ainvoke(
        {"messages": [HumanMessage(content="How are sessions validated?")], "repository_id": "acme/webapp"},
        config={"recursion_limit": 15},
    )

    assert model.call_count == 2
    assert fake_search_service.calls == [("session", 5)]

    answer = final_answer(final_state["messages"])
    assert answer == "Sessions are validated by checking the token."

    sources = collect_search_sources(final_state["messages"])
    assert sources and sources[0]["file_path"] == "app/auth/session.py"

    tool_calls = collect_tool_calls(final_state["messages"])
    assert len(tool_calls) == 1
    assert tool_calls[0].tool == "search_codebase"
    assert tool_calls[0].status == "success"


async def test_read_repository_file_rejects_unknown_path(sample_store, fake_search_service):
    tools = build_tools(search_service=fake_search_service, store=sample_store)
    model = _FakeChatModel(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "read_repository_file", "args": {"file_path": "../../etc/passwd"}, "id": "c1"}],
            ),
            AIMessage(content="That file does not exist in this repository.", tool_calls=[]),
        ]
    )

    graph = build_agent_graph(chat_model=model, tools=tools)
    final_state = await graph.ainvoke(
        {"messages": [HumanMessage(content="read ../../etc/passwd")], "repository_id": "acme/webapp"},
        config={"recursion_limit": 15},
    )

    tool_calls = collect_tool_calls(final_state["messages"])
    assert tool_calls[0].status == "error"


async def test_agent_graph_respects_recursion_limit(sample_store, fake_search_service):
    tools = build_tools(search_service=fake_search_service, store=sample_store)

    class _LoopingModel:
        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages):
            return AIMessage(content="", tool_calls=[{"name": "list_repository_files", "args": {}, "id": "c"}])

    graph = build_agent_graph(chat_model=_LoopingModel(), tools=tools)
    with pytest.raises(GraphRecursionError):
        await graph.ainvoke(
            {"messages": [HumanMessage(content="loop forever")], "repository_id": "acme/webapp"},
            config={"recursion_limit": 6},
        )


def test_chat_response_schema_unchanged():
    """Existing clients reading only answer/sources must still validate fine."""
    response = ChatResponse(answer="hi", sources=[])
    assert response.answer == "hi"
    assert response.sources == []
    assert response.tool_calls == []  # additive field, defaults to empty
