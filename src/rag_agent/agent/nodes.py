
"""
LangGraph nodes for the Deep Learning RAG Interview Prep Agent.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)

from rag_agent.agent.prompts import SYSTEM_PROMPT
from rag_agent.agent.state import AgentResponse, AgentState
from rag_agent.config import LLMFactory, get_settings
from rag_agent.vectorstore.store import VectorStoreManager


@lru_cache(maxsize=1)
def get_llm():
    """Create and cache the configured language model."""
    return LLMFactory(get_settings()).create()


@lru_cache(maxsize=1)
def get_store():
    """Create and cache the ChromaDB vector store."""
    return VectorStoreManager()


def read_state(state, key, default=None):
    """Support both dictionary and attribute-based graph state."""
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def query_rewrite_node(state: AgentState) -> dict:
    """Rewrite the latest user question for vector retrieval."""

    messages = read_state(state, "messages", [])

    human_messages = [
        message
        for message in messages
        if isinstance(message, HumanMessage)
    ]

    original_query = (
        human_messages[-1].content
        if human_messages
        else read_state(state, "original_query", "")
    )

    if not original_query:
        return {
            "original_query": "",
            "rewritten_query": "",
        }

    try:
        llm = get_llm()

        rewrite_prompt = [
            SystemMessage(
                content=(
                    "Rewrite the user's question into a concise "
                    "semantic search query about deep learning. "
                    "Preserve the original meaning and technical terms. "
                    "Return only the rewritten query."
                )
            ),
            HumanMessage(content=original_query),
        ]

        result = llm.invoke(rewrite_prompt)
        rewritten_query = str(result.content).strip()

        if not rewritten_query:
            rewritten_query = original_query

    except Exception:
        # Keep retrieval working if query rewriting fails.
        rewritten_query = original_query

    return {
        "original_query": original_query,
        "rewritten_query": rewritten_query,
    }


def retrieval_node(state: AgentState) -> dict:
    """Retrieve relevant document chunks from ChromaDB."""

    query = read_state(state, "rewritten_query", "")

    if not query:
        query = read_state(state, "original_query", "")

    if not query:
        return {
            "retrieved_chunks": [],
            "no_context_found": True,
        }

    manager = get_store()

    chunks = manager.query(
        query_text=query,
        topic_filter=read_state(state, "topic_filter"),
        difficulty_filter=read_state(state, "difficulty_filter"),
    )

    return {
        "retrieved_chunks": chunks,
        "no_context_found": len(chunks) == 0,
    }


def generation_node(state: AgentState) -> dict:
    """Generate an answer grounded in retrieved documents."""

    chunks = read_state(state, "retrieved_chunks", [])
    no_context_found = read_state(
        state,
        "no_context_found",
        False,
    )

    rewritten_query = read_state(
        state,
        "rewritten_query",
        "",
    )

    if no_context_found or not chunks:
        answer = (
            "I was unable to find relevant information in the "
            "uploaded study materials. Please try a different "
            "question or upload documents covering this topic."
        )

        response = AgentResponse(
            answer=answer,
            sources=[],
            confidence=0.0,
            no_context_found=True,
            rewritten_query=rewritten_query,
        )

        return {
            "final_response": response,
            "messages": [AIMessage(content=answer)],
        }

    context_parts = []
    sources = []
    scores = []

    for chunk in chunks:
        metadata = chunk.metadata

        citation = (
            f"{metadata.topic} | {metadata.source}"
        )

        context_parts.append(
            f"[SOURCE: {citation}]\n"
            f"{chunk.chunk_text}"
        )

        if citation not in sources:
            sources.append(citation)

        scores.append(chunk.score)

    context = "\n\n".join(context_parts)

    confidence = (
        sum(scores) / len(scores)
        if scores
        else 0.0
    )

    original_query = read_state(
        state,
        "original_query",
        "",
    )

    history = read_state(state, "messages", [])

    settings = get_settings()
    llm = get_llm()

    max_tokens = getattr(
        settings,
        "max_context_tokens",
        3000,
    )

    try:
        trimmed_history = trim_messages(
            history,
            max_tokens=max_tokens,
            token_counter="approximate",
            strategy="last",
            start_on="human",
            include_system=False,
        )
    except (ValueError, TypeError):
        trimmed_history = history[-6:]

    # Avoid repeating the current user question.
    if (
        trimmed_history
        and isinstance(trimmed_history[-1], HumanMessage)
    ):
        trimmed_history = trimmed_history[:-1]

    prompt_messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        SystemMessage(
            content=(
                "Answer using only the following retrieved "
                "study material. Cite the relevant source filenames. "
                "If the material does not support a claim, "
                "say that the information is unavailable.\n\n"
                f"{context}"
            )
        ),
        *trimmed_history,
        HumanMessage(content=original_query),
    ]

    result = llm.invoke(prompt_messages)
    answer = str(result.content).strip()

    response = AgentResponse(
        answer=answer,
        sources=sources,
        confidence=confidence,
        no_context_found=False,
        rewritten_query=rewritten_query,
    )

    return {
        "final_response": response,
        "messages": [AIMessage(content=answer)],
    }


def should_retry_retrieval(state: AgentState) -> str:
    """
    Route to generation.

    The generation node handles missing context safely
    without calling the language model.
    """
    return "generate"
