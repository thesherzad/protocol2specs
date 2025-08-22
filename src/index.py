from openai import OpenAI
import numpy as np
import faiss

client = OpenAI()

# Purpose: Convert text chunks (pages, queries, etc.) into dense vectors.
def embed_texts(texts, model):
    # batch to control token limits
    out = []
    for i in range(0, len(texts), 128):
        resp = client.embeddings.create(model=model, input=texts[i:i+128])
        out.extend([d.embedding for d in resp.data])
    return np.array(out, dtype="float32")

# Purpose: Store all protocol chunks in vector space and retrieve the most similar chunks to a query
class LocalIndex:
    def __init__(self, dim): 
        self.index = faiss.IndexFlatIP(dim)
        self.vectors = None
        self.meta = []

    @staticmethod
    def _normalize(v): 
        faiss.normalize_L2(v)
        return v

    def add(self, vecs, metas):
        vecs = self._normalize(vecs.astype("float32"))
        self.index.add(vecs)
        self.vectors = vecs if self.vectors is None else np.vstack([self.vectors, vecs])
        self.meta.extend(metas)

    def search(self, vec, k):
        vec = self._normalize(vec.reshape(1, -1).astype("float32"))
        scores, idxs = self.index.search(vec, k)
        results = []
        for score, i in zip(scores[0], idxs[0]):
            if i == -1: 
                continue
            m = self.meta[i].copy() 
            m["score"] = float(score)
            results.append(m)
        return results

# Purpose: Prepare the FAISS index from the protocol text chunks.
def build_index(chunks, embed_model):
    texts = [c["text"] for c in chunks]
    vecs = embed_texts(texts, embed_model)
    metas = [{"text": c["text"], "page": c["page"], "section": c["section"], "variable_hits": c["variable_hits"]} for c in chunks]
    idx = LocalIndex(vecs.shape[1])
    idx.add(vecs, metas)
    return idx
