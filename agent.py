import re

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from supabase_client import get_vector_store, hybrid_search


SYSTEM_PROMPT = """You are GETY's AI assistant for rubber tree (Hevea brasiliensis) disease management.

LANGUAGE RULE:
- Always reply in the same language the user wrote in.
- If the user writes in Malay, reply in Malay. If English, reply in English.

DISEASE ACCURACY RULE:
- Only answer about the exact disease the user asked about. Never mix diseases.
- If chunks mention multiple diseases, use only the relevant ones.

For any question about rubber tree diseases, call the `retrieve` tool first. Base your answer ONLY on what it returns.

ANSWER LENGTH RULE:
- Match the answer length to the question complexity.
- Simple question (e.g. "what causes Oidium?") → 2-4 sentences max.
- Specific question (e.g. "what fungicide and dosage for Corynespora?") → include the key facts only: fungicide name, rate, interval.
- Only give a full detailed answer if the user explicitly asks for complete information.
- Never dump everything from the chunks — pick only what directly answers the question.

PREVENTION RULE:
- If asked about prevention and no explicit prevention section exists, use fungicide schedule and spray intervals as the answer — proactive application IS prevention for rubber tree diseases.

Formatting:
- Use bullet points only when listing 3 or more items.
- **Bold** key terms: disease names, chemical names.
- No preamble, no repetition, no filler sentences."""


# ── Disease keyword → canonical name map (used for targeted retrieval) ───────
_DISEASE_QUERY_MAP = {
    'oidium':           'Oidium heveae powdery mildew luruhan daun sekunder',
    'powdery mildew':   'Oidium heveae powdery mildew luruhan daun sekunder',
    'colletotrichum':   'Colletotrichum gloeosporioides luruhan daun sekunder',
    'anthracnose':      'Colletotrichum gloeosporioides luruhan daun sekunder',
    'corynespora':      'Corynespora cassiicola luruhan daun sekunder tulang ikan',
    'fusicoccum':       'Fusicoccum leaf blight luruhan daun fusicoccum',
    'leaf blight':      'Fusicoccum leaf blight luruhan daun fusicoccum',
    'phytophthora':     'Phytophthora palmivora abnormal leaf fall',
    'bird eye':         'Bird eye spot bipolaris heveae rintik mata burung',
    'bird\'s eye':      'Bird eye spot bipolaris heveae rintik mata burung',
    'pink disease':     'Pink disease corticium salmonicolor cendawan angin',
    'black stripe':     'Black stripe calar hitam phytophthora botryosa',
    'white root':       'White root disease rigidoporus akar putih',
    'red root':         'Red root disease ganoderma akar merah',
    'brown root':       'Brown root disease phellinus akar perang',
}


def _get_disease_query(question: str) -> str | None:
    """Return a targeted retrieval query if a disease name is detected."""
    q = question.lower()
    for keyword, search_query in _DISEASE_QUERY_MAP.items():
        if keyword in q:
            return search_query
    return None


@tool
def retrieve(query: str) -> str:
    """Search the GETY rubber tree disease knowledge base. Always call this before answering."""
    try:
        targeted_query = _get_disease_query(query)
        search_query = targeted_query if targeted_query else query
        docs = hybrid_search(search_query, k=15)
    except Exception:
        # Fallback to plain vector search if hybrid fails
        docs = get_vector_store().similarity_search(query, k=15)

    if not docs:
        return "No relevant documents found."

    return "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('filename', 'unknown')}, "
        f"chunk {doc.metadata.get('chunk_index', '?')}]\n"
        f"{doc.page_content}"
        for doc in docs
    )


_agent = None


def _get_agent():
    global _agent
    if _agent is None:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
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
