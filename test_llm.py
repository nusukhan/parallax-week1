# ============================================================
# test_llm.py
#
# WHAT THIS FILE DOES:
# A small standalone test to check two things:
#   1. Our OpenRouter API key works
#   2. We can successfully get a response back from the LLM
#
# This was used first (before building the full RAG system) to
# make sure the LLM connection works on its own.
# ============================================================

# The OpenAI library is used to talk to the LLM through OpenRouter
from openai import OpenAI

# ---------- Connect to OpenRouter ----------
# base_url = OpenRouter's address
# api_key  = our private key (replaced with a placeholder for safety)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="PASTE_YOUR_KEY_HERE"
)

# ---------- Send a simple test question to the LLM ----------
# model = which LLM to use (openrouter/free picks a free model)
# messages = the conversation; here just one user message
response = client.chat.completions.create(
    model="openrouter/free",
    messages=[
        {"role": "user", "content": "Say hello in one short sentence."}
    ]
)

# ---------- Print the LLM's reply ----------
# response.choices[0].message.content = the text the LLM sent back
print("LLM says:")
print(response.choices[0].message.content)