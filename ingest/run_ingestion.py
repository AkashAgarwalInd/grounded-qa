import json
import pathlib
import numpy as np
from chunk import fixed_chunks
from embed import embed

MODEL_NAME = "BAAI/bge-small-en-v1.5"
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
    all_chunks = []

    # 1. Load and Chunk
    for file_path in DATA_FILES:
        records = load_jsonl(file_path)
        if not records:
            continue
            
        chunks = fixed_chunks(records, size=512, overlap=64)
        all_chunks.extend(chunks)

    print(f"\n[TOTAL CHUNKS] Generated {len(all_chunks)} chunks.")

    if not all_chunks:
        print("[WARNING] Zero chunks produced. If your text records are shorter than 50 characters, adjust 'len(piece) < 50' in ingest/chunk.py.")
        return

    # 2. Extract Text & Embed
    chunk_texts = [c["text"] for c in all_chunks]
    vectors = embed(chunk_texts, model_name=MODEL_NAME)

    print(f"\n[DONE] Matrix shape: {vectors.shape}")
    print(f"Successfully generated/loaded {len(vectors)} vectors for downstream search.")


if __name__ == "__main__":
    main()