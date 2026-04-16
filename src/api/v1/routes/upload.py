from fastapi import APIRouter, UploadFile, File
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.core.db import get_vector_store
import tempfile
import re

router = APIRouter()


def extract_section(text: str) -> str:
    """
    Extract section title from content.
    Strategy:
    - Look for 'Q: What is XYZ?'
    - Extract 'XYZ'
    """
    match = re.search(r"Q:\s*What is (.+?)\?", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return "General"


@router.post("/admin/upload")
async def upload_file(file: UploadFile = File(...)):
    print("Upload started")

    suffix = file.filename.split('.')[-1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    print("File saved")

    if suffix == "pdf":
        loader = PyPDFLoader(tmp_path)
    elif suffix == "txt":
        loader = TextLoader(tmp_path)
    else:
        return {"error": "Only PDF and TXT supported"}

    docs = loader.load()
    print(f"Loaded {len(docs)} docs")

    for doc in docs:
        doc.metadata["source"] = file.filename
        doc.metadata["page"] = doc.metadata.get("page", 0) + 1
        doc.metadata["section"] = extract_section(doc.page_content)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    vector_store = get_vector_store()
    print("Adding to vector DB...")

    vector_store.add_documents(chunks)

    print("Done storing")

    return {"message": "File processed and stored successfully"}