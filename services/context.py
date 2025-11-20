# services/context.py
from __future__ import annotations
from typing import Any, Dict, List
from dataclasses import dataclass, field

@dataclass
class ContextManager:
    db: "DBAdapter"
    embed_fn: Any                   # callable(text)->vector
    retrieve_fn: Any                # callable(vector,k)->List[(id,score,meta)]
    last_summary: str = ""
    memory_ids: List[str] = field(default_factory=list)

    def remember_json(self, kind: str, obj: Dict[str, Any]):
        text = self._summarize_for_memory(kind, obj)
        mem_id = self.db.add_memory(kind, text, {"kind": kind}, self.embed_fn)
        self.memory_ids.append(mem_id)
        return mem_id

    def rollup_summary(self, delta: Dict[str, Any], llm_summarize):
        self.last_summary = llm_summarize(delta, previous=self.last_summary)
        self.db.write_artifact("summary", {"text": self.last_summary, "delta": delta})
        self.remember_json("summary", {"text": self.last_summary})
        return self.last_summary

    def _summarize_for_memory(self, kind: str, obj: Dict[str, Any]) -> str:
        # very compact; safe to store in FAISS
        return f"[{kind}] " + str({k: obj.get(k) for k in list(obj)[:20]})
