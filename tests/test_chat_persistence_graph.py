"""
Offline tests for Milestone 7's checkpointer integration with the chat agent
graph (app.agents.graph.build_agent_graph).

Uses langgraph.checkpoint.memory.InMemorySaver — not a MongoDB fake — because
it implements the exact same BaseCheckpointSaver interface MongoDBSaver does.
These tests verify *our* usage of that interface (thread_id plumbing, message
restoration/accumulation, isolation between threads) is correct; they are not
testing MongoDB's wire protocol, which is out of scope for the offline suite.

"Persists across simulated application restart": in this app, every request
builds a brand-new graph object (see app.api.routes.chat) — there is no
long-lived graph instance to begin with. The only thing that must survive a
real restart is the checkpointer's backing database. So the correct offline
analogue is: two separate `build_agent_graph(...)` calls, sharing the same
checkpointer instance, must restore the same history — which is exactly what
every real request already relies on, restart or not.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.agents.graph import build_agent_graph
from app.agents.message_utils import final_answer

pytestmark = pytest.mark.anyio


class _FakeChatModel:
    """Returns a fixed reply with no tool calls — persistence tests don't need tool-calling."""

    def __init__(self):
        self.calls: list[list] = []

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        self.calls.append(list(messages))
        turn = sum(1 for m in messages if isinstance(m, HumanMessage))
        return AIMessage(content=f"Reply to turn {turn}", tool_calls=[])


async def test_two_threads_maintain_separate_histories(sample_store, fake_search_service):
    from app.agents.tools import build_tools

    tools = build_tools(search_service=fake_search_service, store=sample_store)
    checkpointer = InMemorySaver()
    model = _FakeChatModel()
    graph = build_agent_graph(chat_model=model, tools=tools, checkpointer=checkpointer)

    await graph.ainvoke(
        {"messages": [HumanMessage(content="Thread 1, message A")], "repository_id": "acme/webapp"},
        config={"configurable": {"thread_id": "thread-1"}},
    )
    await graph.ainvoke(
        {"messages": [HumanMessage(content="Thread 2, message A")], "repository_id": "acme/webapp"},
        config={"configurable": {"thread_id": "thread-2"}},
    )

    state_1 = await graph.aget_state({"configurable": {"thread_id": "thread-1"}})
    state_2 = await graph.aget_state({"configurable": {"thread_id": "thread-2"}})

    texts_1 = [m.content for m in state_1.values["messages"] if isinstance(m, HumanMessage)]
    texts_2 = [m.content for m in state_2.values["messages"] if isinstance(m, HumanMessage)]

    assert texts_1 == ["Thread 1, message A"]
    assert texts_2 == ["Thread 2, message A"]


async def test_existing_thread_restores_previous_messages_on_second_turn(sample_store, fake_search_service):
    from app.agents.tools import build_tools

    tools = build_tools(search_service=fake_search_service, store=sample_store)
    checkpointer = InMemorySaver()
    model = _FakeChatModel()
    graph = build_agent_graph(chat_model=model, tools=tools, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "thread-1"}}

    await graph.ainvoke({"messages": [HumanMessage(content="First question")], "repository_id": "acme/webapp"}, config=config)
    final = await graph.ainvoke({"messages": [HumanMessage(content="Follow-up question")], "repository_id": "acme/webapp"}, config=config)

    human_texts = [m.content for m in final["messages"] if isinstance(m, HumanMessage)]
    assert human_texts == ["First question", "Follow-up question"], "second turn must see the first turn's messages too"

    # No duplication: exactly one AIMessage reply per HumanMessage turn.
    ai_count = sum(1 for m in final["messages"] if isinstance(m, AIMessage))
    assert ai_count == 2

    # The second model call received the full accumulated history (system + both turns).
    second_call_messages = model.calls[-1]
    assert sum(1 for m in second_call_messages if isinstance(m, HumanMessage)) == 2


async def test_rebuilding_the_graph_with_the_same_checkpointer_restores_history(sample_store, fake_search_service):
    """Simulates an app restart: a brand-new graph object, same underlying checkpointer."""
    from app.agents.tools import build_tools

    tools = build_tools(search_service=fake_search_service, store=sample_store)
    checkpointer = InMemorySaver()  # stands in for "the same Mongo database"
    config = {"configurable": {"thread_id": "thread-1"}}

    graph_before_restart = build_agent_graph(chat_model=_FakeChatModel(), tools=tools, checkpointer=checkpointer)
    await graph_before_restart.ainvoke(
        {"messages": [HumanMessage(content="Before restart")], "repository_id": "acme/webapp"}, config=config
    )

    # A brand-new graph instance, as every real request builds — only the
    # checkpointer (standing in for Mongo) is shared.
    graph_after_restart = build_agent_graph(chat_model=_FakeChatModel(), tools=tools, checkpointer=checkpointer)
    state = await graph_after_restart.aget_state(config)

    human_texts = [m.content for m in state.values["messages"] if isinstance(m, HumanMessage)]
    assert human_texts == ["Before restart"]


async def test_without_a_checkpointer_behavior_is_unchanged_from_milestone_5(sample_store, fake_search_service):
    """checkpointer=None (the default) must not persist anything, and must not require one."""
    from app.agents.tools import build_tools

    tools = build_tools(search_service=fake_search_service, store=sample_store)
    model = _FakeChatModel()
    graph = build_agent_graph(chat_model=model, tools=tools)  # no checkpointer passed

    final_state = await graph.ainvoke(
        {"messages": [HumanMessage(content="Hello")], "repository_id": "acme/webapp"}
    )
    assert final_answer(final_state["messages"]) == "Reply to turn 1"

    # No config/thread_id was used, and nothing should be retrievable afterward
    # (there's no checkpointer at all) — aget_state on an uncompiled-with-memory
    # graph raises, confirming no persistence happened.
    with pytest.raises(ValueError):
        await graph.aget_state({"configurable": {"thread_id": "whatever"}})


async def test_extra_context_is_appended_to_system_prompt(sample_store, fake_search_service):
    """Long-term memory injection (extra_context) must reach the model without replacing the base prompt."""
    from langchain_core.messages import SystemMessage

    from app.agents.tools import build_tools

    tools = build_tools(search_service=fake_search_service, store=sample_store)
    model = _FakeChatModel()
    graph = build_agent_graph(
        chat_model=model, tools=tools, extra_context="Known context: this repo uses Flask and Postgres."
    )

    await graph.ainvoke({"messages": [HumanMessage(content="Hi")], "repository_id": "acme/webapp"})

    system_message = next(m for m in model.calls[0] if isinstance(m, SystemMessage))
    assert "DevPilot" in system_message.content  # base prompt still present
    assert "Flask and Postgres" in system_message.content  # injected memory context present
