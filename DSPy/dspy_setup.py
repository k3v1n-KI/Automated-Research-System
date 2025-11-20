# dspy_setup.py
import os
import dspy

# LM backend — swap to your preferred OpenAI model
OPENAI_MODEL = os.getenv("DSPY_OPENAI_MODEL", "gpt-4o-mini")

def init_dspy():
    lm = dspy.OpenAI(model=OPENAI_MODEL)
    dspy.settings.configure(lm=lm, cache=False)  # no local cache: always live
    return lm
