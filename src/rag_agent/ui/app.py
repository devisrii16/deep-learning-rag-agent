
from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage

from rag_agent.agent.graph import get_compiled_graph
from rag_agent.config import get_settings
from rag_agent.corpus.chunker import DocumentChunker
from rag_agent.vectorstore.store import VectorStoreManager


@st.cache_resource
def get_vector_store() -> VectorStoreManager:
    return VectorStoreManager()


@st.cache_resource
def get_chunker() -> DocumentChunker:
    return DocumentChunker()


@st.cache_resource
def get_graph():
    return get_compiled_graph()


def initialise_session_state() -> None:
    defaults = {
        "chat_history": [],
        "ingested_documents": [],
        "selected_document": None,
        "last_ingestion_result": None,
        "thread_id": str(uuid.uuid4()),
        "topic_filter": None,
        "difficulty_filter": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_ingestion_panel(
    store: VectorStoreManager,
    chunker: DocumentChunker,
) -> None:
    st.sidebar.header("📂 Corpus Ingestion")

    uploaded_files = st.sidebar.file_uploader(
        "Upload study materials",
        type=["pdf", "md"],
        accept_multiple_files=True,
    )

    if st.sidebar.button(
        "Ingest Documents",
        disabled=not uploaded_files,
    ):
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_paths = []

                for uploaded_file in uploaded_files:
                    filename = Path(uploaded_file.name).name
                    destination = Path(temp_dir) / filename
                    destination.write_bytes(uploaded_file.getvalue())
                    file_paths.append(destination)

                chunks = chunker.chunk_files(file_paths)
                result = store.ingest(chunks)

            st.session_state.last_ingestion_result = result

            st.sidebar.success(
                f"{result.ingested} chunks added, "
                f"{result.skipped} duplicates skipped."
            )

            for error in result.errors:
                st.sidebar.error(str(error))

        except Exception as exc:
            st.sidebar.error(f"Ingestion failed: {exc}")

    documents = store.list_documents()
    st.session_state.ingested_documents = documents

    st.sidebar.subheader("Stored Documents")

    if not documents:
        st.sidebar.info("Upload .pdf or .md files to populate the corpus.")
        return

    for document in documents:
        source = document["source"]

        with st.sidebar.expander(source):
            st.write(f"Topic: {document['topic']}")
            st.write(f"Chunks: {document['chunk_count']}")

            if st.button("🗑 Remove", key=f"delete_{source}"):
                removed = store.delete_document(source)

                if st.session_state.selected_document == source:
                    st.session_state.selected_document = None

                st.sidebar.success(f"Removed {removed} chunks.")
                st.rerun()


def render_corpus_stats(store: VectorStoreManager) -> None:
    st.sidebar.divider()
    st.sidebar.header("📊 Corpus Statistics")

    stats = store.get_collection_stats()

    st.sidebar.metric("Total Chunks", stats["total_chunks"])

    topics = stats["topics"]
    st.sidebar.write(
        "Topics:",
        ", ".join(topics) if topics else "None",
    )

    if stats["bonus_topics_present"]:
        st.sidebar.success("✅ Bonus topics present")
    else:
        st.sidebar.warning("⚠️ No bonus topics yet")


def render_document_viewer(store: VectorStoreManager) -> None:
    st.subheader("📄 Document Viewer")

    documents = store.list_documents()

    if not documents:
        st.info("Ingest documents using the sidebar to view content here.")
        return

    sources = [document["source"] for document in documents]

    selected = st.session_state.selected_document
    selected_index = sources.index(selected) if selected in sources else 0

    selected_source = st.selectbox(
        "Select document",
        options=sources,
        index=selected_index,
    )

    st.session_state.selected_document = selected_source

    chunks = store.get_document_chunks(selected_source)

    with st.container(height=400):
        for index, chunk in enumerate(chunks, start=1):
            st.markdown(f"**Chunk {index}**")
            st.caption(
                f"Topic: {chunk.metadata.topic} | "
                f"Difficulty: {chunk.metadata.difficulty} | "
                f"Type: {chunk.metadata.type}"
            )
            st.write(chunk.chunk_text)
            st.divider()

    st.caption(f"Total chunks in document: {len(chunks)}")


def render_chat_interface(graph, store: VectorStoreManager) -> None:
    st.subheader("💬 Interview Prep Chat")

    stats = store.get_collection_stats()
    topics = ["All Topics"] + stats["topics"]
    difficulties = [
        "All Levels",
        "beginner",
        "intermediate",
        "advanced",
    ]

    topic_col, difficulty_col = st.columns(2)

    with topic_col:
        selected_topic = st.selectbox(
            "Topic",
            topics,
        )

    with difficulty_col:
        selected_difficulty = st.selectbox(
            "Difficulty",
            difficulties,
        )

    st.session_state.topic_filter = (
        None if selected_topic == "All Topics" else selected_topic
    )
    st.session_state.difficulty_filter = (
        None if selected_difficulty == "All Levels"
        else selected_difficulty
    )

    with st.container(height=400):
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                if message.get("sources"):
                    with st.expander("📎 Sources"):
                        for source in message["sources"]:
                            st.caption(source)

                if message.get("confidence") is not None:
                    st.caption(
                        f"Retrieval confidence: "
                        f"{message['confidence']:.2f}"
                    )

                if message.get("no_context_found"):
                    st.warning(
                        "⚠️ No relevant content found in corpus."
                    )

    query = st.chat_input(
        "Ask about a deep learning topic..."
    )

    if not query:
        return

    st.session_state.chat_history.append(
        {"role": "user", "content": query}
    )

    try:
        agent_input = {
            "messages": [HumanMessage(content=query)],
            "original_query": query,
            "topic_filter": st.session_state.topic_filter,
            "difficulty_filter": st.session_state.difficulty_filter,
        }

        config = {
            "configurable": {
                "thread_id": st.session_state.thread_id
            }
        }

        with st.spinner("Generating answer..."):
            result = graph.invoke(agent_input, config=config)

        response = result.get("final_response")

        if response is None:
            raise ValueError(
                "Agent did not return a final_response."
            )

        if isinstance(response, dict):
            answer = response.get("answer", "")
            sources = response.get("sources", [])
            confidence = response.get("confidence", 0.0)
            no_context_found = response.get(
                "no_context_found", False
            )
        else:
            answer = response.answer
            sources = response.sources
            confidence = response.confidence
            no_context_found = response.no_context_found

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "confidence": confidence,
                "no_context_found": no_context_found,
            }
        )

    except Exception as exc:
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": f"Error generating answer: {exc}",
            }
        )

    st.rerun()


def main() -> None:
    settings = get_settings()

    st.set_page_config(
        page_title=settings.app_title,
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title(f"🧠 {settings.app_title}")
    st.caption(
        "RAG-powered interview preparation — "
        "built with LangChain, LangGraph, and ChromaDB"
    )

    initialise_session_state()

    store = get_vector_store()
    chunker = get_chunker()

    render_ingestion_panel(store, chunker)
    render_corpus_stats(store)

    viewer_col, chat_col = st.columns(
        [1, 1],
        gap="large",
    )

    with viewer_col:
        render_document_viewer(store)

    with chat_col:
        try:
            graph = get_graph()
            render_chat_interface(graph, store)
        except Exception as exc:
            st.error(f"Could not initialise the agent: {exc}")
            st.exception(exc)

if __name__ == "__main__":
    main()
