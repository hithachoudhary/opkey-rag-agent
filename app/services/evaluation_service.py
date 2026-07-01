#Runs your 15 hardcoded automated validation routines using Pydantic structured output judges

import json
from openai import OpenAI
import os
from app.services.retrieval_service import retrieval_service
from app.services.llm_service import llm_service

class EvaluationService:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.test_set = [
            {"question": "What primary functional areas are covered within the Financials offering?", "target": "oracle_financials_implementation_guide.pdf"},
            {"question": "What role is required to opt into features meeting financials business requirements?", "target": "oracle_financials_implementation_guide.pdf"},
            {"question": "What tasks are simplified using the Financials Configuration for Rapid Implementation task list?", "target": "oracle_financials_implementation_guide.pdf"},
            {"question": "Which specific task lists are included for functional areas in rapid implementation?", "target": "oracle_financials_implementation_guide.pdf"},
            {"question": "How do you manage geography setups in the Trading Community Model?", "target": "oracle_financials_implementation_guide.pdf"},
            {"question": "What frameworks are used to implement enterprise structures dynamically?", "target": "oracle_financials_implementation_guide.pdf"},
            {"question": "How are chart of accounts, ledgers, legal entities, and business units uploaded via spreadsheets?", "target": "oracle_financials_implementation_guide.pdf"},
            {"question": "What step must be performed to determine account combinations users can create dynamically?", "target": "oracle_financials_implementation_guide.pdf"},
            {"question": "What is the purpose of the Open First Period task?", "target": "oracle_financials_implementation_guide.pdf"},
            {"question": "How do you define bank branches and accounts using spreadsheets?", "target": "oracle_financials_implementation_guide.pdf"},
            {"question": "What uniform interfaces does Oracle Fusion Tax deliver to business flows?", "target": "oracle_financials_implementation_guide.pdf"},
            {"question": "Which seven report types are accessible via the Financial Reporting Center?", "target": "oracle_financials_implementation_guide.pdf"},
            {"question": "How do you configure the Smart View Client for end users?", "target": "oracle_financials_implementation_guide.pdf"},
            {"question": "What tools are used to create and format reports in the Financial Reporting Center?", "target": "oracle_financials_implementation_guide.pdf"},
            {"question": "What role category are the default roles for segment value security created in?", "target": "oracle_financials_implementation_guide.pdf"}
        ]

    def evaluate_pipeline(self) -> dict:
        results = []
        for item in self.test_set:
            q = item["question"]
            contexts = retrieval_service.get_relevant_context(q, top_k=4)
            hit = 1 if any(item["target"] in c["metadata"]["source"] for c in contexts) else 0
            answer = llm_service.synthesize_answer(q, contexts)
            
            judge_prompt = f"Question: {q}\nContext: {' '.join([c['text'] for c in contexts])}\nAnswer: {answer}\nScore metrics 'faithfulness' and 'relevance' (1 or 0). Return pure JSON keys."
            try:
                res = self.client.chat.completions.create(
                    model="gpt-4o-mini", messages=[{"role": "user", "content": judge_prompt}], response_format={"type": "json_object"}
                )
                scores = json.loads(res.choices[0].message.content)
            except:
                scores = {"faithfulness": 0, "relevance": 0}
                
            results.append({
                "question": q, "hit_rate": hit, "faithfulness": scores.get("faithfulness", 0), "answer_relevance": scores.get("relevance", 0)
            })
        return {
            "avg_hit_rate": sum(r["hit_rate"] for r in results) / len(results),
            "avg_faithfulness": sum(r["faithfulness"] for r in results) / len(results),
            "avg_relevance": sum(r["answer_relevance"] for r in results) / len(results),
            "metrics": results
        }

evaluation_service = EvaluationService()