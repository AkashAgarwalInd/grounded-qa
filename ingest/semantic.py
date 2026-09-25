import re
import numpy as np
from sentence_transformers import SentenceTransformer

# Don't break on '§ 3(1).' or 'No. 22 of 2023.'
SENT = re.compile(r'(?<=[.!?])\s+(?=[A-Z(\u201c])')


def semantic_chunks(record, model, pct=25, max_words=768):
    text = record.get('text', '')
    section = record.get('section', '')

    sents = [s for s in SENT.split(text) if len(s.split()) > 3]
    if len(sents) < 3:
        return [{**record, 'chunker': 'semantic'}]

    v = model.encode(sents, normalize_embeddings=True)
    sim = (v[:-1] * v[1:]).sum(axis=1)  # cosine, already normalised
    thr = np.percentile(sim, pct)  # per-document, not global

    # Section boundary markers that must never be crossed
    sec_break = re.compile(r'(?i)(Sec\.\s+\d+|§\s*\d+|Article\s+\d+)')

    out, buf = [], [sents[0]]
    for s, sc in zip(sents[1:], sim):
        too_long = len(' '.join(buf).split()) > max_words
        crosses_sec = bool(sec_break.search(s))
        if sc < thr or too_long or crosses_sec:
            chunk_text = ' '.join(buf)
            if len(chunk_text.strip()) >= 10:
                out.append({**record, 'text': chunk_text, 'chunker': 'semantic'})
            buf = []
        buf.append(s)
    chunk_text = ' '.join(buf)
    if len(chunk_text.strip()) >= 10:
        out.append({**record, 'text': chunk_text, 'chunker': 'semantic'})
    return out