# app/graph/graph_builder.py
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
from typing import Dict, Any, List, TypedDict

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import tools_condition

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    SystemMessage, AIMessage, ToolMessage, BaseMessage, HumanMessage
)

from app.config import settings
from app.rag.retrieve_tool import retrieve  # @tool(response_format="content_and_artifact")

# ─── 0) State schema: keep chat_id and the user question explicitly ──────────
class AgentState(TypedDict):
    messages: List[BaseMessage]
    chat_id: int
    question: str  # <-- new

# ─── 1) LLM & memory ─────────────────────────────────────────────────────────
llm = ChatOpenAI(
    model="gpt-4.1-nano",
    temperature=0.2,
    openai_api_key=settings.openai_key,
)
memory = MemorySaver()

# ─── helpers ─────────────────────────────────────────────────────────────────
def _latest_human(msgs: List[BaseMessage]) -> str:
    for m in reversed(msgs):
        if getattr(m, "type", None) == "human":
            txt = getattr(m, "content", "") or ""
            if txt.strip():
                return txt.strip()
    return ""

# ─── 2) Node: decide or answer ───────────────────────────────────────────────
def query_or_respond(state: AgentState) -> Dict[str, Any]:
    """
    If general-knowledge → answer directly.
    Else → emit a tool-call to `retrieve`.
    Also ensure state['question'] is set.
    """
    llm_with_tools = llm.bind_tools([retrieve])
    chat_id = state.get("chat_id", "unknown")

    # ensure we carry the question explicitly
    question = state.get("question") or _latest_human(state["messages"])

    system_hint = (
        "You are a study assistant. If a question can be answered from general knowledge, "
        "answer directly and briefly. If it likely needs the user's study materials, "
        f"call the `retrieve` tool and pass the current chat_id ({chat_id}). "
        "Do not fabricate retrieval results."
    )
    fewshots: List[BaseMessage] = [
        SystemMessage(content=system_hint),
        HumanMessage(content="Are most cats nocturnal?"),
        AIMessage(content="Yes. Domestic cats are mostly crepuscular—active at dawn and dusk."),
        HumanMessage(content="Explain theorem 2.3 from my lecture notes."),
        AIMessage(content="I'll consult your lecture materials using the retrieve tool."),
        HumanMessage(content="Are guys usually taller than girls?"),
        AIMessage(content="Yes. Adult males are, on average, taller than adult females worldwide."),
    ]
    msgs = fewshots + state["messages"]

    response = llm_with_tools.invoke(msgs)
    # return the AIMessage *and* the stabilized question back into state
    return {"messages": [response], "question": question}

# ─── 3) Node: run tools (force the real chat_id) ─────────────────────────────
def run_tools(state: AgentState) -> Dict[str, Any]:
    msgs = state["messages"]
    if not msgs or not isinstance(msgs[-1], AIMessage) or not msgs[-1].tool_calls:
        return {"messages": []}

    chat_id = state["chat_id"]
    outs: List[ToolMessage] = []

    for call in msgs[-1].tool_calls:
        if call.get("name") != "retrieve":
            continue

        args = dict(call.get("args", {}) or {})
        args["chat_id"] = chat_id  # hard override to the correct user

        # retrieve must return (content_for_llm, chunks_list)
        try:
            content_for_llm, chunks_list = retrieve.func(**args)
        except Exception as e:
            outs.append(
                ToolMessage(
                    tool_call_id=call["id"],
                    name="retrieve",
                    content=f"[retrieve error] {type(e).__name__}: {e}",
                )
            )
            continue

        # Make the retrieved text visible to the LLM
        if isinstance(chunks_list, list) and chunks_list:
            tool_text = "\n\n".join(str(x) for x in chunks_list)
        else:
            tool_text = (content_for_llm or "").strip()

        outs.append(
            ToolMessage(
                tool_call_id=call["id"],
                name="retrieve",
                content=tool_text,
                artifact=chunks_list,  # raw payload (for inspection/tracing)
            )
        )

    return {"messages": outs}

# ─── 4) Node: final generation ───────────────────────────────────────────────
MAX_CTX_CHARS = 8000

def _gather_context_since_last_toolcall(msgs: List[BaseMessage]) -> str:
    parts: List[str] = []
    for m in reversed(msgs):
        t = getattr(m, "type", None)
        if t == "tool":
            c = getattr(m, "content", "")
            if isinstance(c, str) and c.strip():
                parts.append(c.strip())
            else:
                art = getattr(m, "artifact", None)
                if isinstance(art, list) and art:
                    parts.append("\n\n".join(str(x) for x in art))
                elif isinstance(art, str) and art.strip():
                    parts.append(art.strip())
        elif t == "ai" and getattr(m, "tool_calls", None):
            break
    parts.reverse()
    ctx = "\n\n---\n\n".join(parts)
    if len(ctx) > MAX_CTX_CHARS:
        ctx = ctx[:MAX_CTX_CHARS] + "\n\n[Context truncated]"
    return ctx

def generate(state: AgentState) -> Dict[str, Any]:
    # guaranteed by query_or_respond / handler
    question = (state.get("question") or "").strip()
    context_text = _gather_context_since_last_toolcall(state["messages"])

    sys = (
        "You are a study assistant. Answer ONLY using the CONTEXT below. "
        "If the answer is not in the context, say you don't know and suggest "
        "what the user could upload or clarify. Be concise and structured."
    )
    sys_msg = SystemMessage(
        content=sys
        + (f"\n\nCONTEXT:\n{context_text}" if context_text else "\n\nCONTEXT: [empty]")
        + (f"\n\nQUESTION:\n{question}" if question else "")
    )

    # optional: keep the latest human turn for tone (not required for grounding)
    last_human: List[BaseMessage] = []
    for m in reversed(state["messages"]):
        if getattr(m, "type", None) == "human":
            last_human = [m]
            break

    prompt = [sys_msg, *last_human]
    response = llm.invoke(prompt)
    return {"messages": [response]}

# ─── 5) Build & compile ─────────────────────────────────────────────────────
def build_graph():
    sg = StateGraph(AgentState, config={"memory": memory})
    sg.add_node("decide",   query_or_respond)
    sg.add_node("tools",    run_tools)
    sg.add_node("generate", generate)

    sg.set_entry_point("decide")
    sg.add_conditional_edges("decide", tools_condition, {END: END, "tools": "tools"})
    sg.add_edge("tools", "generate")
    sg.add_edge("generate", END)
    return sg.compile(checkpointer=memory)

graph = build_graph()
