def fixed_chunks(records, size=512, overlap=64):
    """Chunks never cross a section boundary — citations depend on it."""
    out = []
    for r in records:
        text = r.get('text', '')
        words = text.split()
        if not words:
            continue

        # If entire record is short, keep it as a single chunk
        if len(words) <= size:
            if len(text.strip()) >= 10:  # Ignore pure empty whitespace noise
                out.append({**r, 'text': text, 'chunker': 'fixed', 'char_start': 0})
            continue

        step = size - overlap
        for i in range(0, len(words), step):
            piece = ' '.join(words[i:i + size])
            if len(piece) < 50 and i > 0:  # Drop trailing small fragments only if not the first slice
                continue
            out.append({**r, 'text': piece, 'chunker': 'fixed', 'char_start': i})
    return out