"""Talks to Groq through LangChain. We only use it to write the summary at the end.
Scores, anomalies and recommendations are still plain rules. Needs GROQ_API_KEY set."""
import json, os
from functools import lru_cache

from langchain.chat_models import init_chat_model

MODEL = os.getenv("WEALTH_LLM", "groq:openai/gpt-oss-120b")     # Groq's real model id has a slash in it

SYSTEM = (
    "You are a senior wealth advisor writing a quick note to a colleague about a client's review. "
    "Write like a real person talking to a coworker: warm, direct, no corporate jargon, no bullet points, "
    "no headings, no emojis. Start with the client's name and how they're doing overall, then say what needs "
    "attention first and why. Keep it to 3-5 sentences. "
    "Only use the facts in the JSON you're given. Don't make up numbers, don't add advice that isn't listed, "
    "and don't change the priorities. If something is uncertain (like a missing CRM profile), say so plainly."
)


@lru_cache(maxsize=1)
def get_llm():
    return init_chat_model(MODEL, temperature=0.4)


def narrate(facts: dict) -> str:
    resp = get_llm().invoke([("system", SYSTEM), ("human", json.dumps(facts, default=str))])
    return resp.content.strip()
