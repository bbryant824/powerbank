# app/agent/react_agent.py  → replace file contents with this

from __future__ import annotations
import re
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from app.config import settings
from app.rag.retrieve_tool import retrieve


SYSTEM_TEXT = (
    "You are a helpful study assistant. "
    "If a question can be answered from general knowledge, answer briefly. "
    "If the answer likely requires the user's materials, call the `retrieve` tool. "
    "NEVER invent citations. Keep the final answer concise and structured."
)

def build_agent_for_chat(chat_id: int) -> AgentExecutor:
    llm = ChatOpenAI(
        model="gpt-4o-mini",          # tool-calling friendly; use your preferred model
        temperature=0.2,
        openai_api_key=settings.openai_key,
    )

    # Wrap tool so correct chat_id is always applied. Accept a single string input.
    def _retrieve_for_user(user_input: str) -> str:
        # Optional: let users pass "k=6" inline
        k = 4
        q = user_input
        m = re.search(r"\bk\s*=\s*(\d+)\b", user_input)
        if m:
            k = int(m.group(1))
            q = re.sub(r"\bk\s*=\s*\d+\b", "", user_input).strip()

        header, chunks = retrieve.func(question=q, chat_id=chat_id, k=k)
        return header + ("\n\n" + "\n\n".join(chunks) if chunks else "")

    tools = [
        Tool.from_function(
            name="retrieve",
            description=(
                "Retrieve relevant snippets from the user's uploaded PDFs. "
                "Input is the question text; you may append 'k=<int>' to change top-k."
            ),
            func=_retrieve_for_user,
        )
    ]

    # Tools agent prompt: include {tools}; tools agent handles tool routing.

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            SYSTEM_TEXT
            + "\nUse tools only when needed. If no tool is needed, answer directly. "
            "Never invent citations. Keep answers concise and structured."
            # NOTE: removed `{tools}` entirely
        ),
        MessagesPlaceholder("chat_history"),   # list of messages
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm, tools, prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,   # still good practice
        return_intermediate_steps=False,
    )
    return executor
