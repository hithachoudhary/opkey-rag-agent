import os

class Settings:
    CHROMA_DB_PATH: str = "/workspace/chroma_db"
    COLLECTION_NAME: str = "oracle_financials_production_collection"
    
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    LLM_MODEL: str = "gpt-4o-mini"
    
    @property
    def OPENAI_API_KEY(self) -> str:
        return os.getenv("OPENAI_API_KEY", "")

settings = Settings()