import os
import base64
import hashlib
import json
import pathlib
from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.utilities import SQLDatabase
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row


load_dotenv()

_PG_CONNECTION = os.getenv("PG_CONNECTION_STRING")
_PG_DSN = _PG_CONNECTION.replace("postgresql+psycopg://", "postgresql://")

_EMBED_BATCH_SIZE = 50



_embeddings_model = GoogleGenerativeAIEmbeddings(
    model=os.getenv("GOOGLE_EMBEDDING_MODEL"),
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    output_dimensionality=1536
)


def get_vector_store(collection_name: str = "hr_support_desk") -> PGVector:
    return PGVector(
        collection_name=collection_name,
        connection=_PG_CONNECTION,
        embeddings=_embeddings_model,
        use_jsonb=True,
    )


def get_sql_database() -> SQLDatabase:
    """Return a LangChain SQLDatabase connected to the agentic_rag_db (read-only).

    Uses the rag_readonly role from sql/seed.sql — SELECT privileges only.
    Connection string is read from AGENTIC_RAG_DB_URL in the environment.
    """
    db_url = os.getenv("AGENTIC_RAG_DB_URL")
    if not db_url:
        raise ValueError("AGENTIC_RAG_DB_URL is not set. Check your .env file.")
    return SQLDatabase.from_uri(
        db_url,
        include_tables=["billing_statements", "card_transactions", "credit_cards", "customers","reward_transactions"],
        sample_rows_in_table_info=2,
    )


# ---------------------------------------------------------------------------
# Issue 9 fix: Lazy connection pool — reuses existing TCP connections instead
# of opening a new one per request. Created on first use to avoid failing at
# import time when the DB is not yet available (e.g. during tests).
# ---------------------------------------------------------------------------
_pool: ConnectionPool | None = None


def _get_pool() -> ConnectionPool:
    """Return the module-level connection pool, creating it on first call."""
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            _PG_DSN,
            min_size=2,
            max_size=10,
            kwargs={"row_factory": dict_row},
        )
    return _pool


def get_db_conn():
    """Return a pooled connection context manager.

    Usage:
        with get_db_conn() as conn:
            with conn.cursor() as cur: ...
    """
    return _get_pool().connection()


# ---------------------------------------------------------------------------
# Document registry
# ---------------------------------------------------------------------------

def upsert_document(filename: str, source_path: str) -> str:
    """Insert a document record and return its UUID.

    Uses ON CONFLICT so re-ingesting the same filename updates the path
    and returns the existing doc_id rather than creating a duplicate.
    This makes ingestion idempotent at the document level.
    """
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (filename, source_path)
                VALUES (%s, %s)
                ON CONFLICT (filename) DO UPDATE
                    SET source_path = EXCLUDED.source_path,
                        ingested_at  = now()
                RETURNING id
                """,
                (filename, source_path),
            )
            row = cur.fetchone()
        conn.commit()
    return str(row["id"])


# ---------------------------------------------------------------------------
# Chunk storage
# ---------------------------------------------------------------------------
def store_chunks(chunks: list[dict], doc_id: str) -> int:
    print(f"\n Storing {len(chunks)} chunks...")
    if not chunks:
        return 0

    contents = [c["content"] for c in chunks]
    print(" Generating embeddings...")

    # Generate embeddings one-by-one (query style is correct here)
    all_embeddings = []
    for idx, text in enumerate(contents):
        try:
            emb = _embeddings_model.embed_query(text)
            all_embeddings.append(emb)
        except Exception as e:
            print(f"❌ Embedding failed for chunk {idx}: {e}")
            all_embeddings.append(None)

    print(f" Embeddings ready: {len([e for e in all_embeddings if e is not None])}")

    rows_inserted = 0

    with get_db_conn() as conn:
        with conn.cursor() as cur:

            # Remove existing chunks for this document (idempotent ingestion)
            cur.execute(
                "DELETE FROM multimodal_chunks WHERE doc_id = %s::uuid",
                (doc_id,),
            )
            conn.commit()

            for idx, (chunk, embedding) in enumerate(zip(chunks, all_embeddings)):

                # Skip chunks that failed embedding
                if embedding is None:
                    continue

                try:
                    meta = chunk.get("metadata", {})

                    # ✅ Image handling (path already created by docling_parser)
                    image_path = chunk.get("image_path")
                    mime_type = "image/png" if image_path else None

                    embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"

                    cur.execute(
                        """
                        INSERT INTO multimodal_chunks (
                            doc_id,
                            chunk_type,
                            element_type,
                            content,
                            image_path,
                            mime_type,
                            page_number,
                            section,
                            source_file,
                            position,
                            embedding,
                            metadata
                        )
                        VALUES (
                            %s::uuid,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s,
                            %s::jsonb,
                            %s::vector,
                            %s::jsonb
                        )
                        """,
                        (
                            doc_id,
                            chunk.get("content_type"),
                            meta.get("element_type"),
                            chunk.get("content"),
                            image_path,
                            mime_type,
                            meta.get("page_number"),
                            meta.get("section"),
                            meta.get("source_file"),
                            json.dumps(meta.get("position")) if meta.get("position") else None,
                            embedding_str,
                            json.dumps(meta),
                        ),
                    )

                    rows_inserted += 1

                except Exception as e:
                    print(f"❌ Failed inserting chunk {idx}: {e}")
                    continue

        conn.commit()

    print(f"\n Inserted {rows_inserted}/{len(chunks)} chunks")
    return rows_inserted

# ---------------------------------------------------------------------------
# Similarity search
# ---------------------------------------------------------------------------

def similarity_search(
    query: str,
    k: int = 5,
    chunk_type: str | None = None,
) -> list[dict]:
    """Find the k most similar chunks to a natural-language query.

    Args:
        query:      Natural-language question or search string.
        k:          Number of results to return.
        chunk_type: Optional filter — 'text', 'table', or 'image'.

    Returns:
        List of dicts with keys: content, chunk_type, page_number, section,
        source_file, element_type, image_base64, mime_type, position,
        metadata, similarity (0–1 cosine similarity score).

    The <=> operator is pgvector's cosine distance operator.
    Similarity = 1 − cosine_distance, so 1.0 = identical, 0.0 = orthogonal.
    """
    query_vec = _embeddings_model.embed_query(query)  # Issue 8: use singleton
    embedding_str = "[" + ",".join(str(v) for v in query_vec) + "]"

    # Conditionally add a chunk_type filter without SQL injection risk
    # (chunk_type is always passed as a parameterised value, never interpolated)
    type_clause = "AND chunk_type = %(chunk_type)s" if chunk_type else ""

    sql = f"""
        SELECT
            content, chunk_type, page_number, section,
            source_file, element_type, image_path, mime_type,
            position, metadata,
            1 - (embedding <=> %(vec)s::vector) AS similarity
        FROM multimodal_chunks
        WHERE 1=1 {type_clause}
        ORDER BY embedding <=> %(vec)s::vector
        LIMIT %(k)s
    """

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, {"vec": embedding_str, "chunk_type": chunk_type, "k": k})
            rows = cur.fetchall()

    # Read image from filesystem and re-encode as base64 for callers.
    results = []
    for row in rows:
        row = dict(row)
        img_path = row.pop("image_path", None)
        if img_path and os.path.exists(img_path):
            row["image_base64"] = base64.b64encode(
                pathlib.Path(img_path).read_bytes()
            ).decode()
        else:
            row["image_base64"] = None
        results.append(row)

    return results


# ---------------------------------------------------------------------------
# Chunk listing (for preview / debugging)
# ---------------------------------------------------------------------------

def get_all_chunks(chunk_type: str | None = None, limit: int = 200) -> list[dict]:
    """Return all stored chunks, optionally filtered by type.

    Args:
        chunk_type: Optional filter — 'text', 'table', or 'image'.
        limit:      Max rows to return (default 200, safety cap).

    Returns:
        List of dicts with keys: id, content, chunk_type, page_number,
        section, source_file, element_type, image_base64, mime_type,
        position, metadata.
    """
    type_clause = "WHERE chunk_type = %(chunk_type)s" if chunk_type else ""

    sql = f"""
        SELECT
            id, content, chunk_type, page_number, section,
            source_file, element_type, image_path, mime_type,
            position, metadata
        FROM multimodal_chunks
        {type_clause}
        ORDER BY page_number ASC NULLS LAST, id ASC
        LIMIT %(limit)s
    """

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, {"chunk_type": chunk_type, "limit": limit})
            rows = cur.fetchall()

    results = []
    for row in rows:
        row = dict(row)
        img_path = row.pop("image_path", None)
        if img_path and os.path.exists(img_path):
            row["image_base64"] = base64.b64encode(
                pathlib.Path(img_path).read_bytes()
            ).decode()
        else:
            row["image_base64"] = None
        results.append(row)

    return results