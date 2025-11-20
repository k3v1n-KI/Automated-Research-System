# dsp_services.py
from dspy_setup import init_dspy
from dsp_programs import SeedQueryGen, CriticActions, ExtractEntities

class DSPyServices:
    """Holds compiled (or vanilla) DSPy programs."""
    def __init__(self, compiled_seed=None, compiled_critic=None, compiled_extract=None):
        # ensure DSPy configured once
        init_dspy()
        self.seed = compiled_seed or SeedQueryGen()     # can be compiled later
        self.critic = compiled_critic or CriticActions()
        self.extract = compiled_extract or ExtractEntities()

    def gen_seed_queries(self, goal: str, n: int = 12):
        return self.seed(goal=goal, n=n)

    def gen_actions(self, goal: str, round_profile: dict, constraints: dict):
        return self.critic(goal=goal, round_profile=round_profile, constraints=constraints)

    def extract_entities(self, goal: str, page_text: str, schema):
        return self.extract(goal=goal, page_text=page_text, schema=schema)
