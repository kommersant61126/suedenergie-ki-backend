import os
import uuid
from fastapi import FastAPI, UploadFile, HTTPException
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from openai import OpenAI
import pypdf

app = FastAPI(title="Suedenergie KI Backend")

# =========================
# OpenAI
# =========================
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# Qdrant
# =========================
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

COLLECTION_NAME = "suedenergie_docs"
EMBEDDING_SIZE = 1536

# =========================
# Ensure collection exists
# =========================
collections = qdrant.get_collections().collections
if COLLECTION_NAME not in [c.name for c in collections]:
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBEDDING_SIZE,
            distance=Distance.COSINE
        )
    )

# =========================
# Embedding helper
# =========================
def embed_text(text: str):
    emb = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return emb.data[0].embedding

# =========================
# Ingest PDF
# =========================
@app.post("/ingest")
async def ingest_doc(file: UploadFile):
    try:
        reader = pypdf.PdfReader(file.file)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid PDF")

    points = []

    for page in reader.pages:
        text = page.extract_text()
        if not text or not text.strip():
            continue

        points.append({
            "id": str(uuid.uuid4()),
            "vector": embed_text(text),
            "payload": {
                "text": text,
                "source": file.filename
            }
        })

    if not points:
        raise HTTPException(status_code=400, detail="No readable text")

    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    return {"status": "success", "chunks": len(points)}

# =========================
# Chat (RAG)
# =========================
@app.post("/chat")
async def chat(query: str):
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Empty query")

    vector = embed_text(query)

    hits = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        limit=5
    )

    context = "\n\n---\n\n".join(
        hit.payload["text"] for hit in hits
    )

    respons
