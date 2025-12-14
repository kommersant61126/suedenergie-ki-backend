import os
from fastapi import FastAPI, UploadFile
from qdrant_client import QdrantClient
from openai import OpenAI
import uuid
import pypdf

app = FastAPI()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

qdrant = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"]
)

COLLECTION_NAME = "suedenergie_docs"

def embed(text):
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return res.data[0].embedding


@app.post("/ingest")
async def ingest_doc(file: UploadFile):
    reader = pypdf.PdfReader(file.file)
    for page in reader.pages:
        text = page.extract_text()
        if not text:
            continue

        vector = embed(text)

        qdrant.points.upsert(
            collection_name=COLLECTION_NAME,
            points=[{
                "id": str(uuid.uuid4()),
                "vector": vector,
                "payload": {"text": text}
            }]
        )

    return {"status": "success"}


@app.post("/chat")
async def chat(query: str):
    query_vector = embed(query)

    search = qdrant.points.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=5
    )

    context = "\n\n---\n\n".join([hit.payload["text"] for hit in search])

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Du bist die interne KI der Südenergie Photovoltaik GmbH. Antworte professionell und nutze das Firmenwissen."
            },
            {
                "role": "user",
                "content": f"Frage: {query}\n\nFirmenwissen:\n{context}"
            }
        ]
    )

    return {"answer": response.choices[0].message["content"]}
