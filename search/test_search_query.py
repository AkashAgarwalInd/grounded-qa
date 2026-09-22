import sys
import pathlib

# Append project root directory to sys.path
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent))

from qdrant_client import QdrantClient, models as qm
from app.config import settings
from ingest.embed import embed

def query_qdrant(query_text: str, top_k: int = 3):
    client = QdrantClient(url=settings.qdrant_url)
    collection_name = settings.collection_name

    # 1. Embed query vector
    query_vector = embed([query_text], model_name=settings.embed_model)[0]

    # 2. Search using query_points (qdrant-client >= 1.10 API)
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector.tolist(),
        limit=top_k,
        query_filter=qm.Filter(
            must=[
                qm.FieldCondition(
                    key="chunker", 
                    match=qm.MatchValue(value=settings.chunker)
                )
            ]
        )
    )

    print(f"\nQuery: '{query_text}'")
    print(f"Collection: '{collection_name}'")
    print("=" * 60)

    for i, res in enumerate(response.points, 1):
        payload = res.payload
        print(f"[{i}] Score: {res.score:.4f} | Doc: {payload.get('doc_id')} | Section: {payload.get('section')}")
        print(f"    Text snippet: {payload.get('text')[:150]}...\n")

if __name__ == "__main__":
    query_qdrant("What are the duties of a data fiduciary regarding security safeguards?")
    
    
# PYTHONPATH=. uv run python search/test_search_query.py 
# [CACHE HIT] Loaded 1 vectors from .cache/embeddings/9605dd2fe1346f00.npy

# Query: 'What are the duties of a data fiduciary regarding security safeguards?'
# Collection: 'regs_bge-small-en-v1.5_fixed-plain'
# ============================================================
# [1] Score: 0.7799 | Doc: dpdp_act_2023 | Section: Sec. 1
#     Text snippet: consistency. (4) A Data Fiduciary shall implement appropriate technical and organisational measures to ensure effective observance of the provisions o...

# [2] Score: 0.7732 | Doc: gdpr | Section: Section 6
#     Text snippet: 6. The data protection officer may fulfil other tasks and duties. The controller or processor shall ensure that any such tasks and duties do not resul...

# [3] Score: 0.7713 | Doc: dpdp_act_2023 | Section: Sec. 1
#     Text snippet: SEC. 1]
# 9
# (a) the volume and sensitivity of personal data processed;
# (b) risk to the rights of Data Principal;
# (c) potential impact on the sovereignty...