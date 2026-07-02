import logging
from openai import OpenAI
from app.core.config import settings

logger = logging.getLogger("opkey_agent_api")

class LLMService:
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        
        # Frozen production guidelines
        self.system_prompt = (
            "You are an Oracle Financials Enterprise Assistant.\n\n"
            "Rules:\n"
            "1. Answer ONLY using the retrieved documentation.\n"
            "2. Combine information from multiple retrieved chunks into one coherent answer whenever they describe the same topic.\n"
            "3. Return complete lists without omitting items present in the retrieved documentation.\n"
            "4. Never introduce facts that are not supported by the retrieved documentation.\n"
            "5. If the retrieved documentation collectively provides enough information to answer the question, synthesize a complete answer. "
            "Only return \"The provided Oracle documentation does not contain sufficient information to answer this question.\" "
            "when the retrieved documentation is genuinely unrelated or insufficient.\n"
            "6. Format answers using clean Markdown.\n"
            "7. Never mention internal chunk IDs.\n"
            "8. When multiple retrieved chunks complement one another, present a unified answer rather than describing each chunk separately.\n"
            "9. Keep answers concise unless the user explicitly requests a detailed explanation."
        )

    def build_prompt(self, question: str, context_chunks: list) -> str:
        """
        Structures individual retrieved vector database elements into 
        labeled, separated contextual reading passages.
        """
        context_string = "=== RETRIEVED DOCUMENTATION ===\n\n"
        for idx, chunk in enumerate(context_chunks):
            context_string += f"Context {idx + 1} (Page {chunk.get('page')}):\n"
            context_string += f"{chunk.get('excerpt')}\n\n"
            
        context_string += f"=== USER QUESTION ===\n\nQuestion: {question}"
        return context_string

    def generate_answer(self, question: str, context_data: dict) -> dict:
        """Sends the polished context compilation block to gpt-4o-mini for processing."""
        # Baseline safeguard check if citation array is empty
        if not context_data["source_chunks"]:
            return {
                "generated_answer": "The provided Oracle documentation does not contain sufficient information to answer this question.",
                "token_metrics": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }

        full_user_content = self.build_prompt(question, context_data["source_chunks"])
        logger.info(f"Sending context completion payload to model: {settings.LLM_MODEL}")
        
        response = self.client.chat.completions.create(
            model=settings.LLM_MODEL,
            temperature=0.0, # Complete deterministic lock
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": full_user_content}
            ]
        )
        
        generated_text = response.choices[0].message.content
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        
        return {
            "generated_answer": generated_text,
            "token_metrics": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }

llm_service = LLMService()