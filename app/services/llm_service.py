#Synthesizes strictly bounded expert answers using gpt-4o-mini

from openai import OpenAI
import os

class LLMService:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def synthesize_answer(self, query: str, contexts: list) -> str:
        context_str = "\n\n".join([f"[Source: {c['metadata']['source']}, Page: {c['metadata']['page_number']}]: {c['text']}" for c in contexts])
        system_prompt = (
            "You are an expert Oracle Fusion Cloud Financials Architect.\n"
            "Answer the query using ONLY the provided manual context lines below.\n"
            "If the context is insufficient, state that cleanly. Do not hallucinate data configurations.\n"
            "Always cite the exact page numbers contextually in prose.\n\n"
            f"--- CONTEXT ---\n{context_str}"
        )
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content

llm_service = LLMService()