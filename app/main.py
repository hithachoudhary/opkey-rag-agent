from fastapi import FastAPI

app = FastAPI(title="RAG Agent API")

@app.get("/health")
def health_check():
    return {"status": "ok", "docs_indexed": 0}