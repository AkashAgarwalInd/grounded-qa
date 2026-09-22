import hashlib
import pathlib
import numpy as np
from sentence_transformers import SentenceTransformer

# Centralized cache directory for pre-computed vector batches
CACHE = pathlib.Path('.cache/embeddings')
CACHE.mkdir(parents=True, exist_ok=True)


def embed(texts: list[str], model_name: str, batch_size: int = 64) -> np.ndarray:
    """
    Generates normalized embeddings for a list of texts using SentenceTransformers.
    Results are cached to disk (.npy) using a SHA256 hash of (texts + model_name).
    
    Args:
        texts: List of chunked text strings to embed.
        model_name: Name or path of the transformer model (e.g., 'BAAI/bge-small-en-v1.5').
        batch_size: Encoding batch size.
        
    Returns:
        np.ndarray: Array of shape (len(texts), embedding_dim) containing unit vectors.
    """
    if not texts:
        return np.empty((0, 0))

    # Unique cache key derived from content and active model
    cache_key = hashlib.sha256(
        ('\u0000'.join(texts) + model_name).encode('utf-8')
    ).hexdigest()[:16]
    
    cache_path = CACHE / f"{cache_key}.npy"

    # 1. Check cache hit
    if cache_path.exists():
        print(f"[CACHE HIT] Loaded {len(texts)} vectors from {cache_path}")
        return np.load(cache_path)

    # 2. Compute embeddings on cache miss
    print(f"[CACHE MISS] Generating embeddings using '{model_name}'...")
    model = SentenceTransformer(model_name)
    
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,  # Crucial: cosine similarity becomes dot product (a @ b)
        show_progress_bar=True
    )

    # 3. Guard against dimension mismatch before indexing/caching
    expected_dim = model.get_sentence_embedding_dimension()
    assert vecs.shape == (len(texts), expected_dim), (
        f"Shape mismatch: expected ({len(texts)}, {expected_dim}), got {vecs.shape}"
    )

    # 4. Save to disk
    np.save(cache_path, vecs)
    print(f"[CACHE SAVED] Persisted vectors to {cache_path}")
    
    return vecs