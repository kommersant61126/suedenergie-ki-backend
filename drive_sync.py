import os
import io
from datetime import datetime
from googleapiclient.discovery import build
from google.oauth2 import service_account
from pypdf import PdfReader

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import uuid

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")

credentials = service_account.Credentials.from_service_account_file(
    os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
    scopes=SCOPES
)

drive = build("drive", "v3", credentials=credentials)

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

def fetch_drive_files():
    query = f"'{FOLDER_ID}' in parents and mimeType='application/pdf'"
    results = drive.files().list(
        q=query,
        fields="files(id, name, modifiedTime)"
    ).execute()
    return results.get("files", [])

def download_pdf(file_id):
    request = drive.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    fh.write(request.execute())
    fh.seek(0)
    return fh

def extract_text(pdf_bytes):
    reader = PdfReader(pdf_bytes)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def sync_drive_to_qdrant():
    print("🔄 Google Drive Sync gestartet:", datetime.now())
    files = fetch_drive_files()

    for file in files:
        pdf_bytes = download_pdf(file["id"])
        text = extract_text(pdf_bytes)

        if not text.strip():
            continue

        qdrant.upsert(
            collection_name="suedenergie_wissen",
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=[],  # Embeddings kommen aus deinem Ingest
                    payload={
                        "source": "google_drive",
                        "filename": file["name"],
                        "content": text,
                        "modified": file["modifiedTime"]
                    }
                )
            ]
        )

    print("✅ Drive Sync abgeschlossen")
