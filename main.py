import os
import uuid
from fastapi import FastAPI, UploadFile, HTTPException
from qdrant_client import QdrantClient
from openai import OpenAI
import pypdf

app = FastAPI(title="Suedenergie KI Backend")

# =========================
# OpenAI
# =========================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# Qdrant
# =========================
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

COLLECTION_NAME = "suedenergie_docs"

# =========================
# Embeddings
# =========================
def embed_text(text: str):
    emb = client.embeddings.create(
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
        if not text:
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
    if not query:
        raise HTTPException(status_code=400, detail="Empty query")

    query_vector = embed_text(query)

    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=5
    )

    context = "\n\n---\n\n".join(
        [hit.payload["text"] for hit in results]
    )

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "system",
                "content": (
                    "Du bist die interne KI der Südenergie Photovoltaik GmbH. "
                    "Antworte professionell und nutze das Firmenwissen."
                )
            },
            {
                "role": "user",
                "content": f"Frage:\n{query}\n\nFirmenwissen:\n{context}"
            }
        ]
    )

    return {"answer": response.output_text}
