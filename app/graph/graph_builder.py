"""
graph_builder.py  –  Agentic RAG with optional retrieval
─────────────────────────────────────────────────────────
"""

from __future__ import annotations
from typing import Dict, Any, List

from langgraph.graph import StateGraph, MessagesState, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings
from app.vector_store.chroma_client import get_user_collection
from app.rag.retrieve_tool import retrieve

# ─── 2.  Shared objects  ────────────────────────────────────────────────────
llm = ChatOpenAI(
    model_name="gpt-4.1-nano",
    temperature=0.2,
    openai_api_key=settings.openai_key,
)

memory = MemorySaver()      # for checkpoints


# ─── 3. Node-1 : decide or answer  (few-shot version) ──────────────────────

def query_or_respond(state: MessagesState) -> Dict[str, Any]:
    """
    Decide:
      • If the user’s question is general-knowledge or everyday,
        answer directly.
      • Otherwise emit a tool-call to `retrieve`.
    Few-shots guide the model.
    """
    llm_with_tools = llm.bind_tools([retrieve])

    # Few-shot exemplars
    examples = ChatPromptTemplate.from_messages([
        ("system",
         "You are a study assistant. If a question can be answered from "
         "general knowledge or common sense, answer directly. "
         "If it likely needs the user's study materials, call the "
         "`retrieve` tool."),
        # ✱ Example 1 — daily question → direct answer
        ("human",  "Are most cats nocturnal?"),
        ("ai",     "Yes. Domestic cats are naturally crepuscular, "
                   "most active at dawn and dusk."),
        # ✱ Example 2 — study question → tool call
        ("human",  "Explain theorem 2.3 from my lecture notes."),
        ("ai",     "<tool call: retrieve(question='Explain theorem 2.3 "
                   "from my lecture notes.', chat_id=${chat_id})>"),
        # ✱ Example 3 — basic fact → direct answer
        ("human",  "Are guys usually taller than girls?"),
        ("ai",     "Yes. Adult males are on average about 13 cm taller "
                   "than adult females worldwide.")
    ])

    # Current conversation appended after exemplars
    prompt = examples.format_messages() + state["messages"]

    response = llm_with_tools.invoke(prompt)
    return {"messages": [response]}


# ─── 4.  Tool execution node  ───────────────────────────────────────────────
tools_node = ToolNode([retrieve])


# ─── 5.  Node-2 : final generation  ─────────────────────────────────────────
def generate(state: MessagesState) -> Dict[str, Any]:
    """Craft final answer, with or without tool context."""
    # collect newest ToolMessages (if any)
    snippets = [
        m.content for m in state["messages"]
        if m.type == "tool" and m.name == "retrieve"
    ]
    context  = "\n\n".join(snippets)

    system_msg = (
        "You are a helpful study assistant. "
        "If provided, use the context below. "
        "If context is empty or irrelevant, answer from general knowledge."
        f"\n\nContext:\n{context}"
    )

    # keep only human + ai messages (no tool calls)
    convo = [msg for msg in state["messages"]
             if msg.type in ("human", "ai") and not msg.tool_calls]

    prompt = [SystemMessage(system_msg)] + convo
    answer = llm.invoke(prompt)

    return {"messages": [answer]}


# ─── 6.  Build the graph  ───────────────────────────────────────────────────
def build_graph():
    sg = StateGraph(                 # ← REQUIRED positional arg
        MessagesState,               #   tells LangGraph the schema
        config={"memory": memory},    #   optional extras
    )

    sg.add_node("decide",   query_or_respond)
    sg.add_node("tools",    tools_node)
    sg.add_node("generate", generate)

    sg.set_entry_point("decide")
    sg.add_conditional_edges(
        "decide",
        tools_condition,
        {END: END, "tools": "tools"},
    )
    sg.add_edge("tools", "generate")
    sg.add_edge("generate", END)

    return sg.compile(checkpointer=memory)


graph = build_graph()
