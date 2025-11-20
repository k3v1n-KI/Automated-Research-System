# services/usage.py
from dataclasses import dataclass

@dataclass
class Usage:
    tokens: int = 0
    queries: int = 0
    actions: int = 0

    def add_tokens(self, n: int): self.tokens += n
    def add_queries(self, n: int): self.queries += n
    def add_actions(self, n: int): self.actions += n
