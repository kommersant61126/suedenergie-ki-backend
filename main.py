import os
import uuid
from fastapi import FastAPI, UploadFile, HTTPException
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from openai import OpenAI
import pypdf

# ==================================================
# APP
# ==================================================
app = FastAPI(title="Suedenergie KI Backend")

# ==================================================
# OPENAI
# ==================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ==================================================
# QDRANT
# ==================================================
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not QDRANT_URL or not QDRANT_API_KEY:
    raise RuntimeError("QDRANT_URL or QDRANT_API_KEY not set")

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

COLLECTION_NAME = "suedenergie_docs"
EMBEDDING_SIZE = 1536  # text-embedding-3-small

# ==================================================
# ENSURE COLLECTION EXISTS
# ==================================================
existing = [c.name for c in qdrant.get_collections().collections]

if COLLECTION_NAME not in existing:
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBEDDING_SIZE,
            distance=Distance.COSINE
        )
    )

# ==================================================
# EMBEDDING HELPER
# ==================================================
def embed_text(text: str):
    emb = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return emb.data[0].embedding

# ==================================================
# INGEST PDF
# ==================================================
@app.post("/ingest")
async def ingest_doc(file: UploadFile):
    try:
        reader = pypdf.PdfReader(file.file)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid PDF file")

    points = []

    for page in reader.pages:
        text = page.extract_text()
        if not text or not text.strip():
            continue

        vector = embed_text(text)

        points.append({
            "id": str(uuid.uuid4()),
            "vector": vector,
            "payload": {
                "text": text,
                "source": file.filename
            }
        })

    if not points:
        raise HTTPException(status_code=400, detail="No readable text found")

    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    return {
        "status": "success",
        "chunks_indexed": len(points),
        "file": file.filename
    }

# ==================================================
# CHAT (RAG)
# ==================================================
@app.post("/chat")
async def chat(query: str):
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Query is empty")

    try:
        query_vector = embed_text(query)

        results = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=5
        )

        context = "\n\n---\n\n".join(
            hit.payload["text"] for hit in results
        )

        response = openai_client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {
                    "role": "system",
                    "content": (
                        "Du bist die interne KI der Südenergie Photovoltaik GmbH. "
                        "Antworte professionell, sachlich und auf Deutsch. "
                        "Nutze vorrangig das bereitgestellte Firmenwissen. "
                        "Wenn Informationen fehlen, sage das klar."
                    )
                },
                {
                    "role": "user",
                    "content": f"Frage:\n{query}\n\nFirmenwissen:\n{context}"
                }
            ]
        )

        return {
            "answer": response.output_text
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {str(e)}"
        )
