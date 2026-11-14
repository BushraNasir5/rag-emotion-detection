from __future__ import annotations


def rewrite_query(query: str, emotion: str) -> str:
    if emotion == "distress":
        return f"{query} Please prioritize supportive guidance and actionable coping steps."
    if emotion == "positive":
        return f"{query} Provide concise reinforcement and growth-oriented next steps."
    return f"{query} Provide precise factual guidance."
