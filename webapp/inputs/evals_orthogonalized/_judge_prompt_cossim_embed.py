"""Semantic-embedding cosine similarity of judge prompts, with and without the shared evidence gate.

Same entries / gate split as _judge_prompt_cossim.py, but vectors are local sentence embeddings
(all-MiniLM-L6-v2, ONNX export; set MINILM_DIR to a dir with model.onnx + tokenizer.json).
Long texts are split into 256-token windows; the document vector is the token-count-weighted
mean of per-window mean-pooled embeddings, L2-normalized.
"""
import os

import numpy as np
import onnxruntime as ort
from sklearn.metrics.pairwise import cosine_similarity
from tokenizers import Tokenizer

from _judge_prompt_cossim import ROOT, load_entries, save

MODEL_DIR = os.environ.get("MINILM_DIR", os.path.expanduser("~/.cache/openwhispr/embedding-models/all-MiniLM-L6-v2"))
WIN = 256

tok = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
tok.no_truncation()
tok.no_padding()
sess = ort.InferenceSession(os.path.join(MODEL_DIR, "model.onnx"))
CLS, SEP = tok.token_to_id("[CLS]"), tok.token_to_id("[SEP]")


def embed(text):
    ids = tok.encode(text, add_special_tokens=False).ids or [CLS]
    vecs, weights = [], []
    for s in range(0, len(ids), WIN - 2):
        chunk = np.array([[CLS] + ids[s:s + WIN - 2] + [SEP]], dtype=np.int64)
        out = sess.run(None, {"input_ids": chunk, "attention_mask": np.ones_like(chunk),
                              "token_type_ids": np.zeros_like(chunk)})[0]
        vecs.append(out[0].mean(axis=0))
        weights.append(chunk.shape[1])
    w = np.array(weights, dtype=np.float32)
    doc = (np.stack(vecs) * w[:, None]).sum(0) / w.sum()
    return doc / np.linalg.norm(doc)


if __name__ == "__main__":
    entries = load_entries()
    labels = [e[0] for e in entries]
    S_with = cosine_similarity(np.stack([embed(e[1]) for e in entries]))
    S_without = cosine_similarity(np.stack([embed(e[2]) for e in entries]))
    save(labels, S_with, os.path.join(ROOT, "_judge_cossim_with_preamble_embed.csv"))
    save(labels, S_without, os.path.join(ROOT, "_judge_cossim_without_preamble_embed.csv"))
    iu = np.triu_indices_from(S_with, k=1)
    print(f"{len(entries)} metric prompts; mean off-diag with={S_with[iu].mean():.3f} without={S_without[iu].mean():.3f}")
