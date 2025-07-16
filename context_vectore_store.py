# context_vector_store.py

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class ContextVectorStore:
    def __init__(self):
        self.model  = SentenceTransformer("all-MiniLM-L6-v2")
        try:
            res = faiss.StandardGpuResources()            # create GPU resource pool
            cpu_index = faiss.IndexFlatL2(384)            # base (CPU) index
            self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        except Exception as e:
            print(f"Failed to initialize GPU resources: {e}")
            # Fallback to CPU index if GPU is not available
            print("Using CPU index instead.")
            self.index  = faiss.IndexFlatL2(384)
                
        self.stored = []  # list of (summary_text, metadata)

    def add(self, summary_text: str, metadata: dict = None):
        """
        Embed a summary and add to the FAISS index.
        """
        vec = self.model.encode(summary_text)
        self.index.add(np.array([vec], dtype="float32"))
        self.stored.append((summary_text, metadata or {}))

    def query(self, text: str, top_k: int = 5) -> list[str]:
        """
        Embed `text` and return the top_k most relevant summary_texts.
        """
        vec = self.model.encode(text)
        D, I = self.index.search(np.array([vec], dtype="float32"), top_k)
        return [self.stored[i][0] for i in I[0] if i < len(self.stored)]
