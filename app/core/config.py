import os

class Settings:
    # Read straight from environment variables with safe fallbacks
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    CHROMA_DB_PATH: str = "/workspace/chroma_db"
    COLLECTION_NAME: str = "oracle_financials_collection"

settings = Settings()