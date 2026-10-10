
"""
chunker.py
==========
Document loading and chunking pipeline.

Converts PDF and Markdown files into structured DocumentChunk
objects for embedding and storage in ChromaDB.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from rag_agent.agent.state import ChunkMetadata, DocumentChunk
from rag_agent.config import Settings, get_settings
from rag_agent.vectorstore.store import VectorStoreManager


class DocumentChunker:
    """Load and split PDF and Markdown documents."""

    DEFAULT_CHUNK_SIZE = 512
    DEFAULT_CHUNK_OVERLAP = 50

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def chunk_file(
        self,
        file_path: Path,
        metadata_overrides: dict | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> list[DocumentChunk]:
        """Load one file and convert it into document chunks."""

        file_path = Path(file_path)

        if not file_path.is_file():
            raise FileNotFoundError(
                f"Document not found: {file_path}"
            )

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError(
                "chunk_overlap must be nonnegative and "
                "smaller than chunk_size"
            )

        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            raw_chunks = self._chunk_pdf(
                file_path, chunk_size, chunk_overlap
            )
        elif suffix in (".md", ".markdown"):
            raw_chunks = self._chunk_markdown(
                file_path, chunk_size, chunk_overlap
            )
        else:
            raise ValueError(
                f"Unsupported file type: {suffix}. "
                "Supported formats: PDF and Markdown."
            )

        metadata = self._infer_metadata(
            file_path, metadata_overrides
        )

        chunks = []

        for raw_chunk in raw_chunks:
            text = raw_chunk["text"].strip()

            if not text:
                continue

            chunk_id = VectorStoreManager.generate_chunk_id(
                file_path.name, text
            )

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    chunk_text=text,
                    metadata=metadata,
                )
            )

        logger.info(
            "Created {} chunks from '{}'",
            len(chunks),
            file_path.name,
        )

        return chunks

    def chunk_files(
        self,
        file_paths: list[Path],
        metadata_overrides: dict | None = None,
    ) -> list[DocumentChunk]:
        """Chunk multiple files, logging individual file errors."""

        all_chunks = []

        for file_path in file_paths:
            try:
                chunks = self.chunk_file(
                    file_path,
                    metadata_overrides=metadata_overrides,
                )
                all_chunks.extend(chunks)

            except Exception as exc:
                logger.error(
                    "Failed to chunk '{}': {}",
                    file_path,
                    exc,
                )

        return all_chunks

    def _chunk_pdf(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[dict]:
        """Extract PDF pages and split their text into chunks."""

        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import (
            RecursiveCharacterTextSplitter,
        )

        loader = PyPDFLoader(str(file_path))
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        raw_chunks = []

        for page in pages:
            page_number = page.metadata.get("page", 0) + 1

            text_chunks = splitter.split_text(
                page.page_content
            )

            for text in text_chunks:
                if text.strip():
                    raw_chunks.append(
                        {
                            "text": text,
                            "page": page_number,
                        }
                    )

        return raw_chunks

    def _chunk_markdown(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[dict]:
        """Split Markdown by headings and then by chunk size."""

        from langchain_text_splitters import (
            MarkdownHeaderTextSplitter,
            RecursiveCharacterTextSplitter,
        )

        markdown_text = file_path.read_text(
            encoding="utf-8"
        )

        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ],
            strip_headers=False,
        )

        sections = header_splitter.split_text(
            markdown_text
        )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        raw_chunks = []

        for section in sections:
            header = (
                section.metadata.get("Header 3")
                or section.metadata.get("Header 2")
                or section.metadata.get("Header 1")
                or ""
            )

            text_chunks = text_splitter.split_text(
                section.page_content
            )

            for text in text_chunks:
                if text.strip():
                    raw_chunks.append(
                        {
                            "text": text,
                            "header": header,
                        }
                    )

        return raw_chunks

    def _infer_metadata(
        self,
        file_path: Path,
        overrides: dict | None = None,
    ) -> ChunkMetadata:
        """Infer metadata from the filename and apply overrides."""

        file_path = Path(file_path)

        parts = file_path.stem.split("_")

        difficulty_levels = {
            "beginner",
            "intermediate",
            "advanced",
        }

        difficulty = "intermediate"
        topic_parts = parts

        if parts[-1].lower() in difficulty_levels:
            difficulty = parts[-1].lower()
            topic_parts = parts[:-1]

        topic = " ".join(topic_parts).strip() or "General"

        bonus_topics = {
            "som",
            "self organizing map",
            "self organizing maps",
            "boltzmannmachine",
            "boltzmann machine",
            "gan",
            "generative adversarial network",
        }

        metadata_values = {
            "topic": topic,
            "difficulty": difficulty,
            "type": "concept",
            "source": file_path.name,
            "related_topics": [],
            "is_bonus": (
                topic.lower().replace("-", " ").replace("_", " ")
                in bonus_topics
            ),
        }

        if overrides:
            metadata_values.update(overrides)

        # ChromaDB requires scalar metadata values.
        # ChunkMetadata.to_dict() handles storage serialization.
        return ChunkMetadata(**metadata_values)
