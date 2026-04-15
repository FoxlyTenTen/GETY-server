import re

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from supabase_client import get_vector_store


SYSTEM_PROMPT = """You are GETY's AI assistant for rubber tree (Hevea brasiliensis) leaf disease management.

For any question about rubber tree diseases, treatments, symptoms, or farm management, you MUST call the `retrieve` tool first to search the knowledge base. Base your answer ONLY on what the tool returns.

If the retrieved context does not contain enough information to answer, say so honestly. Never fabricate facts.

Formatting rules — follow these strictly:
- Write in clean, practical prose.
- Use short paragraphs. Use markdown bullet lists only when listing 3 or more distinct items.
- Use **bold** sparingly for key terms (disease names, chemicals, actions).
- Do NOT include inline source citations like "(filename.pdf, chunk 12)" — source documents are displayed separately to the user as chips, so inline citations are redundant and clutter the answer.
- Do NOT mention chunk numbers, document filenames, or say "according to the document".
- Keep answers focused and concise — no preamble, no repetition."""


@tool
def retrieve(query: str) -> str:
    """Search the GETY knowledge base for information relevant to the query."""
    vector_store = get_vector_store()
    docs = vector_store.similarity_search(query, k=5)
    if not docs:
        return "No relevant documents found."
    return "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('filename', 'unknown')}, "
        f"chunk {doc.metadata.get('chunk_index', '?')}]\n{doc.page_content}"
        for doc in docs
    )


_agent = None


def _get_agent():
    global _agent
    if _agent is None:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
        _agent = create_agent(
            model=llm,
            tools=[retrieve],
            system_prompt=SYSTEM_PROMPT,
        )
    return _agent


def _convert_history(chat_history: list[dict]) -> list:
    messages = []
    for msg in chat_history[-6:]:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


_SOURCE_HEADER = re.compile(r"\[Source:\s*([^,]+),\s*chunk\s*(\d+)\]")


def _extract_sources(messages: list) -> list[dict]:
    sources: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        for match in _SOURCE_HEADER.finditer(content):
            filename = match.group(1).strip()
            chunk_index = int(match.group(2))
            key = (filename, chunk_index)
            if key in seen:
                continue
            seen.add(key)
            sources.append({"filename": filename, "chunk_index": chunk_index})
    return sources


def _extract_answer(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            content = msg.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = [
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                ]
                joined = "".join(parts).strip()
                if joined:
                    return joined
    return ""


def query_rag(question: str, chat_history: list[dict]) -> dict:
    agent = _get_agent()
    history = _convert_history(chat_history)
    history.append(HumanMessage(content=question))

    result = agent.invoke({"messages": history})
    messages = result.get("messages", [])

    return {
        "answer": _extract_answer(messages),
        "sources": _extract_sources(messages),
    }
