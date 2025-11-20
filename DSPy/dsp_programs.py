# dsp_programs.py
import dspy
from dsp_signatures import SeedQueryGenSig, CriticActionsSig, ExtractEntitiesSig

class SeedQueryGen(dspy.Module):
    def __init__(self):
        super().__init__()
        self.gen = dspy.Predict(SeedQueryGenSig)

    def forward(self, goal: str, n: int = 4):
        # Always generate exactly 4 queries (original will be prepended by planner)
        out = self.gen(goal=goal, n=4)
        queries = out.queries or []
        # Clean and dedupe, ensure exactly 4
        clean_qs = [str(q).strip() for q in queries if str(q).strip() and str(q).strip().lower() != goal.lower()]
        seen = set()
        unique_qs = []
        for q in clean_qs:
            if q not in seen:
                seen.add(q)
                unique_qs.append(q)
        while len(unique_qs) < 4:
            unique_qs.append(f"{goal} alternative {len(unique_qs)+1}")
        return unique_qs[:4]

class CriticActions(dspy.Module):
    def __init__(self):
        super().__init__()
        self.crit = dspy.Predict(CriticActionsSig)

    def forward(self, goal: str, round_profile: dict, constraints: dict):
        out = self.crit(goal=goal, round_profile=round_profile, constraints=constraints)
        return out.actions_json or "{}"

class ExtractEntities(dspy.Module):
    def __init__(self):
        super().__init__()
        self.ext = dspy.Predict(ExtractEntitiesSig)

    def forward(self, goal: str, page_text: str, schema: list[str]):
        out = self.ext(goal=goal, page_text=page_text, schema=schema)
        return out.items or []
