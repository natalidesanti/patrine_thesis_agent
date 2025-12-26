"""
Preprocess script:
- Extracts text from PDF
- Builds sliding-window chunks
- Computes embeddings
- Saves everything to disk
"""

#------------------
# Libraries
#------------------

import sys
import re
import json
import numpy as np
from pathlib import Path
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer

#------------------
# Cleaning and sliding
#------------------

def clean_text(text: str) -> str:
    
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def sliding_window_chunks(text, window = 600, stride = 300, min_words = 50):
    
    words = text.split()
    chunks = []
    for i in range(0, len(words), stride):
        chunk = " ".join(words[i:i + window])
        if len(chunk.split()) >= min_words:
            chunks.append(chunk)
    return chunks

#------------------
# Main
#------------------

def main(pdf_path: str, out_dir: str):
    
    out = Path(out_dir)
    out.mkdir(parents = True, exist_ok = True)

    print(f"Reading thesis PDF: {pdf_path}")
    raw_text = clean_text(extract_text(pdf_path))

    print("Chunking text")
    chunks = sliding_window_chunks(raw_text)

    print(f"Embedding {len(chunks)} chunks")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(chunks, convert_to_numpy = True, show_progress_bar = True, normalize_embeddings = True).astype(np.float32)

    print("Saving artifacts")
    np.save(out / "embeddings.npy", embeddings)
    with open(out / "chunks.json", "w", encoding = "utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    config = {
        "pdf": pdf_path,
        "num_chunks": len(chunks),
        "embedding_model": "all-MiniLM-L6-v2",
        "window": 600,
        "stride": 300,
    }
    with open(out / "config.json", "w", encoding = "utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Preprocessing complete. Data saved to {out}")


if __name__ == "__main__":
    
    if len(sys.argv) != 3:
        
        print("Usage: python preprocess_thesis.py <thesis.pdf> <output_dir>")
        sys.exit(1)
        
    main(sys.argv[1], sys.argv[2])
