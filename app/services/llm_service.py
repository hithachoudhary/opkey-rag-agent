import os

class LLMService:
    def __init__(self):
        self.system_prompt = (
            "You are an expert Enterprise Solutions Architect specializing in Oracle Fusion Cloud Financials.\n"
            "Your objective is to answer user implementation questions with absolute factual precision using ONLY the provided context.\n\n"
            "CRITICAL OPERATIONAL RULES:\n"
            "1. Base your answer entirely on the verified context segments provided below.\n"
            "2. If the context does not contain sufficient facts to answer the question, state explicitly: "
            "'I am sorry, but the provided documentation does not contain enough information to fulfill this request.'\n"
            "3. Do not assume, extrapolate, or inject outside knowledge of Oracle systems.\n"
            "4. Keep your formatting highly professional, using clear bullet points where applicable."
        )

    def build_prompt(self, question: str, context_string: str) -> str:
        """
        Structures the raw token input combining the system guidelines, 
        retrieved document facts, and user query.
        """
        return (
            f"=== SYSTEM ROLE ===\n{self.system_prompt}\n\n"
            f"=== RETRIEVED CONTEXT FROM MANUAL ===\n{context_string}\n\n"
            f"=== USER QUERY ===\nQuestion: {question}\n\n"
            f"Final Precision Answer:"
        )

    def generate_answer(self, question: str, context_data: dict) -> dict:
        """
        Compiles the context payload and prepares the generative completion call.
        Includes a clean mock fallback mechanism while awaiting account top-up.
        """
        full_prompt = self.build_prompt(question, context_data["context_string"])
        
        # TODO: Wire up actual OpenAI chat completion client when credits are live:
        # response = client.chat.completions.create(
        #     model="gpt-4o-mini",
        #     messages=[{"role": "user", "content": full_prompt}]
        # )
        
        # MOCK IMPLEMENTATION FOR STAGE-BY-STAGE VALIDATION
        mock_answer = (
            f"Based on the extracted documentation, the implementation configuration for your request is fully structured. "
            f"(This is an architectural placeholder response for query: '{question}'). "
            f"To complete this configuration, verify the enterprise structure parameters on sheet {context_data['source_chunks'][0]['page_number']}."
        )

        return {
            "prompt_compiled_successfully": True,
            "tokens_estimated": int(len(full_prompt) / 4),
            "generated_answer": mock_answer,
            "sources_used": context_data["source_chunks"]
        }

llm_service = LLMService()