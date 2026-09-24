import numpy as np
import json
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient, models as qm
from app.config import settings
import pathlib
from loguru import logger


def build_bm25_corpus(chunks: list[dict]) -> list[list[str]]:
    """
    Build BM25 corpus from chunks.
    Each chunk's text is tokenized into a list of tokens.
    BM25 works best with lowercase tokens and simple preprocessing.
    """
    corpus = []
    for chunk in chunks:
        text = chunk.get("text", "")
        if not text:
            corpus.append([])
            continue
        # Simple tokenization: lowercase, split on whitespace, remove pure punctuation
        tokens = text.lower().split()
        # Filter out tokens that are pure punctuation
        tokens = [t for t in tokens if t.isalnum() or (t.isalpha() and len(t) > 2)]
        corpus.append(tokens)
    return corpus


def index_chunks_to_qdrant(chunks: list[dict], vecs: np.ndarray):
    """
    Indexes chunks and dense vectors into Qdrant using configuration from app.config.
    
    Collection name is derived dynamically from settings.collection_name:
    e.g., 'regs_bge-small-en-v1.5_fixed-plain'
    """
    client = QdrantClient(url=settings.qdrant_url)
    collection_name = settings.collection_name

    logger.info(f"[INDEXING] Connecting to Qdrant at {settings.qdrant_url}")
    logger.info(f"[INDEXING] Target collection: '{collection_name}'")

    # 1. Re-create collection matching vector dimension
    vector_dim = vecs.shape[1] if len(vecs) > 0 else 384
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=qm.VectorParams(size=vector_dim, distance=qm.Distance.COSINE),
    )

    # 2. Payload indexes are required for filtered searches (chunker and doc_id)
    for field in ("chunker", "doc_id"):
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field,
            field_schema=qm.PayloadSchemaType.KEYWORD,
        )

    # 3. Deterministic integer IDs preserve idempotency across re-runs
    points = [
        qm.PointStruct(id=i, vector=vecs[i].tolist(), payload=chunks[i])
        for i in range(len(chunks))
    ]

    # 4. Batch upsert
    batch_size = 256
    for i in range(0, len(points), batch_size):
        client.upsert(
            collection_name=collection_name, 
            points=points[i:i + batch_size], 
            wait=True
        )

    # 5. Build and save BM25 corpus to disk for later retrieval
    corpus = build_bm25_corpus(chunks)
    cache_dir = pathlib.Path(".cache/bm25")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"bm25_{collection_name}.pkl"
    import pickle
    with open(cache_path, "wb") as f:
        pickle.dump({"chunks": chunks, "corpus": corpus}, f)
    logger.info(f"[BM25] Corpus saved to {cache_path}")

    # 5. Verify total count matches input chunk size
    indexed_count = client.count(collection_name=collection_name).count
    if indexed_count != len(chunks):
        logger.error(f"Mismatch: expected {len(chunks)} points, but Qdrant has {indexed_count}")
        raise ValueError(f"Qdrant indexing mismatch: expected {len(chunks)}, got {indexed_count}")
    else:
        logger.success(f"Collection '{collection_name}' successfully indexed {indexed_count} points.")