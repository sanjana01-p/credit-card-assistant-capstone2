import os
import json
import re
import base64

from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse

from src.api.v1.schemas.query_schema import QueryRequest
from src.api.v1.services.query_service import query_documents
from src.api.v1.agents.agent import (
    rag_graph,
    stream_final_answer,
    query_with_vision
)
from src.core.db import similarity_search
from src.ingestion.ingestion import run_ingestion

router = APIRouter()

UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# Upload Endpoint
# ---------------------------------------------------------------------
@router.post("/admin/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    run_ingestion(file_path)
    return {"file": file.filename, "message": "Upload and embedding successful"}


# ---------------------------------------------------------------------
# Normal (non-streaming) Query
# ---------------------------------------------------------------------
@router.post("/query")
def query_endpoint(request: QueryRequest):
    return query_documents(request.query)


# ---------------------------------------------------------------------
# Streaming Query (Text + SQL + Image + Vision)
# ---------------------------------------------------------------------
@router.post("/query/stream")
async def query_stream(request: QueryRequest):

    async def event_generator():

        # --------------------------------------------------
        # 1. Run LangGraph RAG / SQL agent
        # --------------------------------------------------
        initial_state = {
            "query": request.query,
            "hyde_query": "",
            "use_hyde": "",
            "retrieved_docs": [],
            "reranked_docs": [],
            "response": {},
            "retries": 0,
            "messages": [{"role": "user", "content": request.query}],
            "route": "",
            "generated_sql": "",
            "sql_result": "",
        }

        final_state = rag_graph.invoke(initial_state)

        #  Extract ONLY answer text
        response_obj = final_state.get("response")

        if isinstance(response_obj, dict):
            final_answer = response_obj.get("answer", "")
        else:
            final_answer = str(response_obj)

        generated_sql = final_state.get("generated_sql")
        sql_result = final_state.get("sql_result")
        reranked_docs = final_state.get("reranked_docs", []) or []

        # Normalize SQL result
        if isinstance(sql_result, dict):
            sql_result = json.dumps(sql_result, indent=2)
        elif not isinstance(sql_result, str):
            sql_result = str(sql_result)

        has_sql = bool(generated_sql or sql_result)
        has_docs = bool(reranked_docs)

      

        if final_answer and not reranked_docs:
            yield json.dumps({
                "type": "text",
                "content": final_answer
            }) + "\n"

            yield json.dumps({
                "type": "chunks",
                "content": [{
                    "content": f"SQL executed:\n{generated_sql}\n\nResult:\n{sql_result}",
                    "source": "agentic_rag_db",
                    "page": "N/A",
                }]
            }) + "\n"

            return 


        
        if final_answer and (has_sql or has_docs):

            # Answer only (clean text)
            yield json.dumps({
                "type": "text",
                "content": final_answer
            }) + "\n"

            chunks_payload = []

            # RAG chunks
            for doc in reranked_docs:
                chunks_payload.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "N/A"),
                    "page": doc.metadata.get("page", "N/A"),
                })

            # SQL chunk
            if has_sql:
                chunks_payload.append({
                    "content": f"SQL executed:\n{generated_sql}\n\nResult:\n{sql_result}",
                    "source": "agentic_rag_db",
                    "page": "N/A",
                })

            if chunks_payload:
                yield json.dumps({
                    "type": "chunks",
                    "content": chunks_payload
                }) + "\n"

            return

        # --------------------------------------------------
        # 2. Detect IMAGE intent
        # --------------------------------------------------
        image_keywords = ["image", "picture", "figure", "chart", "diagram", "photo"]
        explain_keywords = ["explain", "describe", "analyze", "interpret", "what does"]

        q = request.query.lower()
        is_image_query = any(k in q for k in image_keywords)
        is_explain_image = is_image_query and any(k in q for k in explain_keywords)

        page_match = re.search(r"page\s*(\d+)", q)
        requested_page = int(page_match.group(1)) if page_match else None

        # --------------------------------------------------
        # 3. FORCE IMAGE RETRIEVAL
        # --------------------------------------------------
        forced_image_docs = []

        if is_image_query:
            image_rows = similarity_search(
                query=request.query,
                k=3,
                chunk_type="image"
            )

            for row in image_rows:
                if requested_page and row.get("page_number") != requested_page:
                    continue

                img_b64 = None
                img_path = row.get("image_path")
                print(img_path)
                if img_path and os.path.exists(img_path):
                    with open(img_path, "rb") as f:
                        img_b64 = base64.b64encode(f.read()).decode()

                forced_image_docs.append({
                    "image_base64": img_b64,
                    "image_path": row.get("image_path",""),

                    "page": row.get("page_number", "N/A"),
                    "source": row.get("source_file", "N/A"),
                    "mime_type": row.get("mime_type", "image"),
                })

        forced_image_docs = forced_image_docs[:1]
       
        if is_explain_image and forced_image_docs:
            context = "\n\n".join(doc.page_content for doc in reranked_docs)

            vision_answer = query_with_vision(
                query=request.query,
                context=context,
                image_base64=forced_image_docs[0]["image_base64"]
            )

            yield json.dumps({
                "type": "text",
                "content": vision_answer
            }) + "\n"

            for img in forced_image_docs:
                yield json.dumps({
                    "type": "image",
                    # "content": img["image_base64"],
                    "image_path": img["image_path"],
                    "page": img["page"],
                    "source": img["source"],
                    "mime_type": img["mime_type"],
                }) + "\n"
            return

        if is_image_query and forced_image_docs:
            yield json.dumps({
                "type": "text",
                "content": "Below is the requested image from the document."
            }) + "\n"

            for img in forced_image_docs:
                yield json.dumps({
                    "type": "image",
                    # "content": img["image_base64"],
                    "page": img["page"],
                    "source": img["source"],
                    "image_path": img["image_path"],
                    "mime_type": img["mime_type"],
                }) + "\n"
            return

        # --------------------------------------------------
        # 4. NORMAL RAG TEXT STREAMING
        # --------------------------------------------------
        context = "\n\n".join(doc.page_content for doc in reranked_docs)

        async for token in stream_final_answer(context, request.query):

            if isinstance(token, dict):
                token = token.get("content") or token.get("answer") or ""

            if not isinstance(token, str):
                token = str(token)

            yield json.dumps({
                "type": "text",
                "content": token
            }) + "\n"

        # --------------------------------------------------
        # 5. Send Retrieved Chunks
        # --------------------------------------------------
        chunks_payload = []
        for doc in reranked_docs:
            chunks_payload.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "N/A"),
                "page": doc.metadata.get("page", "N/A"),
            })

        yield json.dumps({
            "type": "chunks",
            "content": chunks_payload
        }) + "\n"

    return StreamingResponse(
        event_generator(),
        media_type="application/x-ndjson"
    )