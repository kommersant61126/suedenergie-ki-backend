import os
import uuid
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

from apscheduler.schedulers.background import BackgroundScheduler

from openai import OpenAI

# ==================================================
# ENV
# ==================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# ==================================================
# APP
# ==================================================

app = FastAPI(
    title="Suedenergie KI Backend",
    description="Interne Wissens-KI für Südenergie Photovoltaik GmbH",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # später auf Vercel-Domain einschränken
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================================================
# OPENAI (optional lokal)
# ==================================================

openai_client = None
if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
else:
    print("⚠️ OPENAI_API_KEY fehlt – Chat läuft im Fallback-Modus")

# ==================================================
# QDRANT
# ==================================================

if not QDRANT_URL or not QDRANT_API_KEY:
    raise RuntimeError("QDRANT_URL oder QDRANT_API_KEY fehlt")

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

COLLECTION_NAME = "suedenergie_firmenwissen"
VECTOR_SIZE = 1536

if COLLECTION_NAME not in [c.name for c in qdrant.get_collections().collections]:
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )

# ==================================================
# MODELS
# ==================================================

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str

# ==================================================
# EMBEDDING
# ==================================================

def embed_text(text: str) -> List[float]:
    if not openai_client:
        return [0.0] * VECTOR_SIZE

    emb = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return emb.data[0].embedding

# ==================================================
# ROUTES
# ==================================================

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "Suedenergie KI Backend",
    }

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query ist leer")

    query_vector = embed_text(req.query)

    hits = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=5,
    )

    context = "\n\n---\n\n".join(
        hit.payload.get("text", "") for hit in hits
    )

    if not openai_client:
        return ChatResponse(
            answer="⚠️ KI ist noch nicht aktiv konfiguriert (OPENAI_API_KEY fehlt)."
        )

    response = openai_client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "system",
                "content": (
                    "Du bist die interne KI der Südenergie Photovoltaik GmbH. "
                    "Antworte professionell, sachlich und auf Deutsch. "
                    "Nutze primär das Firmenwissen. "
                    "Wenn etwas nicht bekannt ist, sage das klar."
                ),
            },
            {
                "role": "user",
                "content": f"Frage:\n{req.query}\n\nFirmenwissen:\n{context}",
            },
        ],
    )

    return ChatResponse(answer=response.output_text)

# ==================================================
# GOOGLE DRIVE AUTO SYNC (STUB)
# ==================================================

def drive_sync_job():
    if not GOOGLE_DRIVE_FOLDER_ID or not GOOGLE_APPLICATION_CREDENTIALS:
        print("⚠️ Google Drive Sync übersprungen (ENV fehlt)")
        return

    print("🔄 Google Drive Auto-Sync läuft")
    # 👉 Hier kommt als Nächstes:
    # - Dateien listen
    # - PDFs / Docs lesen
    # - Embeddings erzeugen
    # - Qdrant upsert

# ==================================================
# SCHEDULER
# ==================================================

scheduler = BackgroundScheduler()
scheduler.add_job(drive_sync_job, "interval", minutes=10)
scheduler.start()

@app.on_event("shutdown")
def shutdown_event():
    scheduler.shutdown()
