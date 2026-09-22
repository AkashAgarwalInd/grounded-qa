import numpy as np
from qdrant_client import QdrantClient, models as qm
from app.config import settings


def index_chunks_to_qdrant(chunks: list[dict], vecs: np.ndarray):
    """
    Indexes chunks and dense vectors into Qdrant using configuration from app.config.
    
    Collection name is derived dynamically from settings.collection_name:
    e.g., 'regs_bge-small-en-v1.5_fixed-plain'
    """
    client = QdrantClient(url=settings.qdrant_url)
    collection_name = settings.collection_name

    print(f"[INDEXING] Connecting to Qdrant at {settings.qdrant_url}")
    print(f"[INDEXING] Target collection: '{collection_name}'")

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

    # 5. Verify total count matches input chunk size
    indexed_count = client.count(collection_name=collection_name).count
    assert indexed_count == len(chunks), (
        f"Mismatch: expected {len(chunks)} points, but Qdrant has {indexed_count}"
    )

    print(f"[SUCCESS] Collection '{collection_name}' successfully indexed {indexed_count} points.")