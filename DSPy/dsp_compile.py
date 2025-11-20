# dsp_compile.py
import dspy
from dsp_objectives import seed_query_objective, critic_objective, extract_objective
from dsp_programs import SeedQueryGen, CriticActions, ExtractEntities

def compile_seed_query(train_examples):
    program = SeedQueryGen()
    tele = dspy.teleprompt.BootstrapFewShot(metric=seed_query_objective, max_bootstrapped_demos=8)
    return tele.compile(program, trainset=train_examples)

def compile_critic(train_examples):
    program = CriticActions()
    tele = dspy.teleprompt.BootstrapFewShot(metric=critic_objective, max_bootstrapped_demos=8)
    return tele.compile(program, trainset=train_examples)

def compile_extractor(train_examples):
    program = ExtractEntities()
    tele = dspy.teleprompt.BootstrapFewShot(metric=extract_objective, max_bootstrapped_demos=6)
    return tele.compile(program, trainset=train_examples)
