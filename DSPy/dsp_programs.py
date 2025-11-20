# dsp_programs.py
import dspy
from dsp_signatures import SeedQueryGenSig, CriticActionsSig, ExtractEntitiesSig

class SeedQueryGen(dspy.Module):
    def __init__(self):
        super().__init__()
        self.gen = dspy.Predict(SeedQueryGenSig)

    def forward(self, goal: str, n: int = 12):
        out = self.gen(goal=goal, n=n)
        return out.queries or []

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
