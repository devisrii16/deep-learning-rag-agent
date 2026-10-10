
"""
store.py
========
ChromaDB vector store management.

Handles persistent storage, ingestion, duplicate detection,
retrieval, document inspection, and deletion.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import chromadb
from loguru import logger

from rag_agent.agent.state import (
    ChunkMetadata,
    DocumentChunk,
    IngestionResult,
    RetrievedChunk,
)
from rag_agent.config import EmbeddingFactory, Settings, get_settings


class VectorStoreManager:
    """Manage a persistent ChromaDB collection."""

    BATCH_SIZE = 100

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._embeddings = EmbeddingFactory(self._settings).create()
        self._client = None
        self._collection = None
        self._initialise()

    def _initialise(self) -> None:
        """Create or connect to the persistent ChromaDB collection."""
        try:
            db_path = Path(self._settings.chroma_db_path)
            db_path.mkdir(parents=True, exist_ok=True)

            self._client = chromadb.PersistentClient(path=str(db_path))
            self._collection = self._client.get_or_create_collection(
                name=self._settings.chroma_collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            logger.info(
                "ChromaDB collection '{}' initialized with {} chunks",
                self._settings.chroma_collection_name,
                self._collection.count(),
            )
        except Exception as exc:
            logger.exception("ChromaDB initialization failed")
            raise RuntimeError(
                "Could not initialize ChromaDB"
            ) from exc

    @staticmethod
    def generate_chunk_id(source: str, chunk_text: str) -> str:
        """Generate a deterministic chunk ID."""
        content = f"{source}::{chunk_text}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    def check_duplicate(self, chunk_id: str) -> bool:
        """Check whether a chunk already exists."""
        result = self._collection.get(ids=[chunk_id])
        return chunk_id in result["ids"]

    def ingest(self, chunks: list[DocumentChunk]) -> IngestionResult:
        """Embed and store document chunks, skipping duplicates."""
        result = IngestionResult()

        for start in range(0, len(chunks), self.BATCH_SIZE):
            batch = chunks[start:start + self.BATCH_SIZE]
            pending = []

            for chunk in batch:
                try:
                    if self.check_duplicate(chunk.chunk_id):
                        result.skipped += 1
                    else:
                        pending.append(chunk)
                except Exception as exc:
                    result.errors.append(
                        f"{chunk.chunk_id}: {exc}"
                    )
                    logger.error(
                        "Duplicate check failed for {}: {}",
                        chunk.chunk_id,
                        exc,
                    )

            if not pending:
                continue

            try:
                embeddings = self._embeddings.embed_documents(
                    [chunk.chunk_text for chunk in pending]
                )

                self._collection.upsert(
                    ids=[chunk.chunk_id for chunk in pending],
                    embeddings=embeddings,
                    documents=[chunk.chunk_text for chunk in pending],
                    metadatas=[
                        chunk.metadata.to_dict() for chunk in pending
                    ],
                )

                result.ingested += len(pending)
                result.document_ids.extend(
                    chunk.chunk_id for chunk in pending
                )

            except Exception as exc:
                logger.error("Batch ingestion failed: {}", exc)
                result.errors.extend(
                    f"{chunk.chunk_id}: {exc}" for chunk in pending
                )

        logger.info(
            "Ingestion complete: {} ingested, {} skipped, {} errors",
            result.ingested,
            result.skipped,
            len(result.errors),
        )
        return result

    def query(
        self,
        query_text: str,
        k: int | None = None,
        topic_filter: str | None = None,
        difficulty_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve relevant chunks using cosine similarity."""
        k = k or self._settings.retrieval_k

        if k < 1 or self._collection.count() == 0:
            return []

        filters = []

        if topic_filter:
            filters.append({"topic": topic_filter})

        if difficulty_filter:
            filters.append({"difficulty": difficulty_filter})

        if len(filters) == 2:
            where_filter = {"$and": filters}
        elif len(filters) == 1:
            where_filter = filters[0]
        else:
            where_filter = None

        query_embedding = self._embeddings.embed_query(query_text)

        query_args = {
            "query_embeddings": [query_embedding],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }

        if where_filter is not None:
            query_args["where"] = where_filter

        results = self._collection.query(**query_args)

        retrieved = []

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for chunk_id, text, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            score = 1.0 - float(distance)

            if score < self._settings.similarity_threshold:
                continue

            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    chunk_text=text,
                    metadata=ChunkMetadata.from_dict(metadata),
                    score=score,
                )
            )

        retrieved.sort(key=lambda chunk: chunk.score, reverse=True)
        return retrieved

    def list_documents(self) -> list[dict]:
        """List unique source documents and their chunk counts."""
        results = self._collection.get(include=["metadatas"])
        documents = {}

        for metadata in results["metadatas"]:
            if not metadata:
                continue

            source = metadata.get("source", "unknown")
            topic = metadata.get("topic", "unknown")

            if source not in documents:
                documents[source] = {
                    "source": source,
                    "topic": topic,
                    "chunk_count": 0,
                }

            documents[source]["chunk_count"] += 1

        return sorted(
            documents.values(),
            key=lambda item: item["source"],
        )

    def get_document_chunks(self, source: str) -> list[DocumentChunk]:
        """Retrieve all chunks belonging to one source document."""
        results = self._collection.get(
            where={"source": source},
            include=["documents", "metadatas"],
        )

        chunks = []

        for chunk_id, text, metadata in zip(
            results["ids"],
            results["documents"],
            results["metadatas"],
        ):
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    chunk_text=text,
                    metadata=ChunkMetadata.from_dict(metadata),
                )
            )

        return chunks

    def get_collection_stats(self) -> dict:
        """Return collection statistics."""
        results = self._collection.get(include=["metadatas"])
        metadatas = [
            item for item in results["metadatas"] if item
        ]

        topics = sorted({
            item.get("topic", "unknown")
            for item in metadatas
        })

        sources = sorted({
            item.get("source", "unknown")
            for item in metadatas
        })

        bonus_topics_present = any(
            str(item.get("is_bonus", "false")).strip().lower() == "true"
            for item in metadatas
        )

        return {
            "total_chunks": self._collection.count(),
            "topics": topics,
            "sources": sources,
            "bonus_topics_present": bonus_topics_present,
        }

    def delete_document(self, source: str) -> int:
        """Delete all chunks from a source document."""
        results = self._collection.get(
            where={"source": source}
        )

        chunk_ids = results["ids"]
        count = len(chunk_ids)

        if chunk_ids:
            self._collection.delete(ids=chunk_ids)

        logger.info(
            "Deleted {} chunks from source '{}'",
            count,
            source,
        )
        return count
