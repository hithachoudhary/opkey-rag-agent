import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    CHROMA_DB_PATH: str = "/workspace/chroma_db"
    COLLECTION_NAME: str = "oracle_financials_collection"
    PORT: int = 8000

settings = Settings()