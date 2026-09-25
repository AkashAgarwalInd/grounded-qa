import json
import pathlib
import numpy as np
from app.config import settings
from .chunk import fixed_chunks
from .embed import embed
from .index_chunks_to_qdrant import index_chunks_to_qdrant
from .semantic import semantic_chunks
from sentence_transformers import SentenceTransformer

DATA_FILES = [
    "data/clean/ccpa_civ_code_downloaded.jsonl",
    "data/clean/dpdp_act_2023.jsonl",
    "data/clean/gdpr.jsonl",
]


def load_jsonl(file_path: str) -> list[dict]:
    """Loads JSONL records into a list of dicts."""
    records = []
    path = pathlib.Path(file_path)

    if not path.exists():
        print(f"[SKIP] File not found: {file_path}")
        return records

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                print(f"[ERR] Invalid JSON on line {line_num} in {file_path}: {e}")

    print(f"[LOADED] {len(records)} records from {file_path}")
    return records


def main():
    # Step 1: Load raw JSONL records from all data sources
    all_records = []
    for file_path in DATA_FILES:
        records = load_jsonl(file_path)
        all_records.extend(records)

    print(f"\n[TOTAL RECORDS] Loaded {len(all_records)} raw records.")

    if not all_records:
        print("[ERROR] No records found to process. Stopping.")
        return

    # Step 2: Chunk documents based on configured chunker strategy
    model = SentenceTransformer(settings.embed_model)
    if settings.chunker == 'semantic':
        all_chunks = semantic_chunks(all_records, model=model, pct=25, max_words=768)
    else:
        all_chunks = fixed_chunks(all_records, size=512, overlap=64)
    print(f"[CHUNKED] Generated {len(all_chunks)} text chunks.")

    if not all_chunks:
        print("[ERROR] Zero chunks produced. Please check your raw text records.")
        return

    # Step 3: Extract texts and generate/fetch cached dense embeddings
    chunk_texts = [c["text"] for c in all_chunks]
    
    # embed() automatically checks .cache/embeddings/ based on content hash + model_name
    vectors = embed(chunk_texts, model_name=settings.embed_model)
    print(f"[VECTORS READY] Matrix shape: {vectors.shape}")

    # Step 4: Strict validation check before indexing
    assert len(all_chunks) == vectors.shape[0], (
        f"Length Mismatch: {len(all_chunks)} chunks vs {vectors.shape[0]} vectors."
    )

    # Step 5: Index vectors and chunk metadata directly into Qdrant
    print(f"\n[QDRANT INDEX] Target collection: '{settings.collection_name}'")
    index_chunks_to_qdrant(all_chunks, vectors)


if __name__ == "__main__":
    main()