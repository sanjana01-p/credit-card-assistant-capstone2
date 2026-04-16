import os
from typing import TypedDict, List, Literal

import cohere
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langchain_core.runnables.graph import MermaidDrawMethod
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from typing import Annotated
from langgraph.graph.message import add_messages
from src.api.v1.tools.fts_search_tool import fts_search
from src.api.v1.tools.vector_search_tool import query_documents
from src.api.v1.tools.hybrid_search_tool import hybrid_search
from src.api.v1.schemas.query_schema import AIResponse
from pydantic import BaseModel
from langchain.messages import HumanMessage

from src.api.v1.tools.tools import RAGState
from src.core.db import get_sql_database



load_dotenv(override=True)

os.environ["PYPPETEER_CHROMIUM_REVISION"] = "1263111"



def _get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_LLM_MODEL"),
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

def generate_hyde_query(query: str) -> str:
    print("\n[HyDE] Fallback triggered — generating hypothetical answer")

    llm = _get_llm()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Write a concise, factual answer to the user's question. "
            "This text will be used ONLY to improve document retrieval."
        ),
        ("human", "{query}")
    ])

    response = (prompt | llm).invoke({"query": query})
    content = response.content

    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )

    hyde_text = content.strip()


    print("[HyDE] Hypothetical document:")
    print("----- HyDE TEXT START -----")
    print(hyde_text)
    print("----- HyDE TEXT END -------")

    return hyde_text

def is_image_query(query) -> bool:
    if isinstance(query, list):
        query = " ".join(str(q) for q in query)

    if isinstance(query, dict):
        query = str(query)

    if not isinstance(query, str):
        return False

    q = query.lower()

    return any(word in q for word in [
        "image", "picture", "figure", "diagram", "photo", "chart"
    ])


def query_with_vision(query: str, context: str, image_base64: str):
    """
    Uses Gemini Pro Vision to answer a question using BOTH text context and image.
    """

    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_LLM_MODEL"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    # Multimodal prompt: text + image
    content_parts = [
        {
            "type": "text",
            "text": (
                "You are a financial analyst assistant. "
                "Use the provided document context AND the image to answer the question.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}"
            )
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image_base64}"
            }
        }
    ]

    message = HumanMessage(content=content_parts)
    response = llm.invoke([message])

    return response.content

# --- STREAMING SUPPORT (ADD ONLY, DO NOT MODIFY EXISTING CODE) ---

from typing import AsyncGenerator

async def stream_final_answer(context: str, query: str) -> AsyncGenerator[str, None]:
    """
    Streams the LLM answer token-by-token.
    This does NOT affect LangGraph execution.
    """
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_LLM_MODEL"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        streaming=True,
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant. Answer the user's question using the provided context."
        ),
        ("human", "Context:\n{context}\n\nQuestion: {query}")
    ])

    chain = prompt | llm

    async for chunk in chain.astream({
        "context": context,
        "query": query
    }):
        if chunk.content:
            yield chunk.content


# ── Node 0: Router ────────────────────────────────────────────────────────────
# Uses Gemini (structured output) to classify the user's query.
#
# "product" → query is about products, prices, stock, orders, categories
#             → routes to nl2sql_node (PostgreSQL / agentic_rag_db)
# "document" → query is about policies, procedures, text documents
#             → routes to the RAG pipeline (vector_search → rerank → generate_answer)

class _RouteDecision(BaseModel):
    route: Literal["product", "document", "hybrid"]
    reason: str


def router_node(state: RAGState) -> RAGState:
    llm = _get_llm()
    structured_llm = llm.with_structured_output(_RouteDecision)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a query router for an agentic RAG system.
            Classify the user's query into EXACTLY one of two routes:

            "product"  — the query asks about products, product prices, stock/inventory,
                        product categories, customer orders, order items, or anything
                        answerable from a structured e-commerce database with tables:
                        products, categories, orders, order_items.

            "document" — the query asks about policies, procedures, guidelines,
                        regulations, or any topic that requires reading text documents.

            "hybrid" - the query asks about both the products, product prices, stock/inventory,
                        product categories, customer orders, order items, or anything
                        answerable from a structured e-commerce database with tables:
                        products, categories, orders, order_items and also the query asks about policies, procedures, guidelines,
                        regulations, or any topic that requires reading text documents.

            Reply with the route and a one-sentence reason."""
        ),
        ("human", "Query: {query}")
    ])

    chain = prompt | structured_llm
    decision = chain.invoke({"query": state["query"]})
    print(f"[router_node] Route → '{decision.route}' | Reason: {decision.reason}")
    return {**state, "route": decision.route}

def hybrid_node(state: RAGState) -> RAGState:
    print("[hybrid_node] Running hybrid product + document pipeline")

    # ── Product side (NL2SQL) ────────────────────────────────────────
    product_state = nl2sql_node(state)
    sql_response = product_state.get("response", {})
    generated_sql = product_state.get("generated_sql", "")
    sql_result = product_state.get("sql_result", "")

    # ── Document side (RAG) ──────────────────────────────────────────
    doc_state = state
    doc_state = search_agent_node(doc_state)
    doc_state = search_result_node(doc_state)
    doc_state = rerank_node(doc_state)


    return {
        **state,
        "hybrid_sql_response": sql_response,
        "generated_sql": generated_sql,
        "sql_result": sql_result,
        "reranked_docs": doc_state.get("reranked_docs", []),
        "response": "hybrid_ready"
    }


# ── Node NL2SQL: Translate query to SQL → Execute → Summarise ─────────────────
# Step 1  create_sql_query_chain generates a safe SELECT statement using the
#          live DB schema (table/column names + 2 sample rows per table).
# Step 2  SQLDatabase.run() executes the SQL on the read-only rag_readonly user.
#          Even if the LLM hallucinated a DML statement, the DB role blocks it.
# Step 3  Gemini summarises the raw results as a structured AIResponse.

def nl2sql_node(state: RAGState) -> RAGState:
    llm = _get_llm()
    db = get_sql_database()

    # ── Step 1: Generate SQL using Gemini + live schema ─────────────────────
    schema_info = db.get_table_info()

    sql_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a PostgreSQL expert. Given the database schema below, 
            write a single valid SELECT query that answers the user's question.

            Rules:
            - Return ONLY the raw SQL — no explanation, no markdown fences, no backticks.
            - Use only the tables and columns present in the schema.
            - Do NOT generate INSERT, UPDATE, DELETE, DROP, or any DML/DDL statements.
            - Always add a LIMIT clause (max 50 rows) unless the question asks for aggregates.
            - For product or text searches: NEVER search for the full multi-word phrase as one
            ILIKE pattern. Instead, split the search into individual meaningful keywords
            and OR them together across both name and description columns.
            Example — user asks "wireless headset":
                WHERE (name ILIKE '%wireless%' OR description ILIKE '%wireless%')
                OR (name ILIKE '%headset%'  OR description ILIKE '%headset%')
                OR (name ILIKE '%headphones%' OR description ILIKE '%headphones%')
            Use your knowledge of synonyms (headset/headphones, laptop/notebook, etc.)
            to cast a wider net when the exact term may not match.

            Database schema:
            {schema}"""
        ),
        ("human", "Question: {question}")
    ])

    sql_chain = sql_prompt | llm
    raw_sql = sql_chain.invoke({
        "schema": schema_info,
        "question": state["query"]
    })
    # Gemini may return content as a list of parts or a plain string
    content = raw_sql.content
    if isinstance(content, list):
        content = "".join(
            p.get("text", "") if isinstance(p, dict) else str(p)
            for p in content
        )
    generated_sql = content.strip().strip("```").strip()
    if generated_sql.lower().startswith("sql"):
        generated_sql = generated_sql[3:].strip()
    print(f"[nl2sql_node] Generated SQL:\n{generated_sql}")

    # ── Step 2: Execute SQL ──────────────────────────────────────────────────
    try:
        sql_result: str = db.run(generated_sql)
    except Exception as exc:
        sql_result = f"SQL execution error: {exc}"
    print(f"[nl2sql_node] Raw result (truncated): {str(sql_result)[:200]}")

    # ── Step 3: Summarise into AIResponse ────────────────────────────────────
    structured_llm = llm.with_structured_output(AIResponse)
    answer_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful data analyst. Answer the user's question using "
            "the SQL query results below. Be concise and format numbers/lists clearly. "
            "Set policy_citations to empty string, "
            "page_no to 'N/A', and document_name to 'agentic_rag_db'."
        ),
        (
            "human",
            "Question: {query}\n\n"
            "SQL Used:\n{sql}\n\n"
            "Query Results:\n{result}"
        )
    ])

    chain = answer_prompt | structured_llm
    answer = chain.invoke({
        "query": state["query"],
        "sql": generated_sql,
        "result": sql_result
    })
    print("[nl2sql_node] Answer generated.")
    response = answer.model_dump()
    response["policy_citations"] = "N/A"
    response["sql_query_executed"] = generated_sql
    return {
        **state,
        "generated_sql": generated_sql,
        "sql_result": str(sql_result),
        "response": response
    }



@tool
def fts_search_tool(query: str, k: int = 20):
    """Full-text keyword search over internal documents"""
    return fts_search(query, k)

@tool
def hybrid_search_tool(query: str, k: int = 20):
    """Hybrid search tool over internal documents"""
    return hybrid_search(query, k)

@tool
def vector_search_tool(query: str, k: int = 20):
    """Semantic vector similarity search"""
    return query_documents(query, k)

search_tools = [fts_search_tool, vector_search_tool, hybrid_search_tool]


def search_agent_node(state: RAGState) -> RAGState:
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_LLM_MODEL"),
        google_api_key=os.getenv("GOOGLE_API_KEY")
    ).bind_tools(search_tools)

    response = llm.invoke([
        {
            "role": "system",
            "content": (
                "You are a retrieval agent. Choose the best search strategy:\n"
                "- Use `fts_search_tool` only if the query contains exact policy names, acronyms, specific document titles\n"
                "- Use `vector_search_tool` for 'latest', 'current', 'explain', 'how', 'what is',semantic meaning\n"
                "- Use `hybrid_search_tool` if unsure or query is complex\n"
                "Return only tool calls."
            )
        },
        {"role": "user", "content": state["query"]}
    ])

    if hasattr(response, "tool_calls") and response.tool_calls:
        for tool_call in response.tool_calls:
            print(
                f"[search_agent_node] Tool selected: {tool_call['name']}"
                f"| args: {tool_call['args']}"
            )
    else:
        print("[search_agent_node] No tool selected by agent")
    return {
        **state,
        "retrieved_docs": [],
        "response": response,
        "messages": [response]
    }

def should_continue_search(state: RAGState) -> str:
    last = state['response']
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END 


def search_result_node(state: RAGState) -> RAGState:
    print("\n[search_result_node] Collecting tool results")

    query = state.get("query", "")
    if isinstance(query, list):
        query = " ".join(str(q) for q in query)

    if is_image_query(state["query"]):
        print("[search_result_node] 🖼️ Image query detected")

        from src.core.db import similarity_search

        image_rows = similarity_search(
            query=state["query"],
            k=5,
            chunk_type="image"
        )

        print(f"[search_result_node] Retrieved {len(image_rows)} image chunks")

        docs = []
        for row in image_rows:
            docs.append(
                Document(
                    page_content=row.get("content", "Image"),
                    metadata={
                        "image_base64": row["image_base64"],
                        "mime_type": row["mime_type"],
                        "page": row["page_number"],
                        "source": row["source_file"],
                        "chunk_type": "image",
                    }
                )
            )

        return {
            **state,
            "retrieved_docs": docs,
            "response": "image_retrieved"
        }

    docs = []

    # Step 1: Collect docs returned by ToolNode
    for msg in state["messages"]:
        if isinstance(msg.content, list):
            docs.extend(msg.content)

    print(f"[search_result_node] Docs from tool execution: {len(docs)}")

    # Step 2: If docs found → return immediately
    if docs:
        print("[search_result_node] ✅ Using documents returned by tool")
        return {**state, "retrieved_docs": docs[:100]}

    # Step 3: No docs found → trigger HyDE fallback
    print("[search_result_node] ⚠️ No documents retrieved from tool")
    print("[search_result_node] 🔁 Triggering HyDE-based hybrid search fallback")

    # Generate HyDE query
    hyde_query = generate_hyde_query(state["query"])

    # Perform hybrid search using HyDE
    hyde_docs = hybrid_search(hyde_query, k=25)

    print(f"[search_result_node] HyDE hybrid search returned {len(hyde_docs)} documents")

    if not hyde_docs:
        print("[search_result_node] ❌ HyDE fallback also returned 0 documents")
    else:
        print("[search_result_node] ✅ Using HyDE-retrieved documents")

    return {
        **state,
        "hyde_query": hyde_query,
        "use_hyde": True,
        "retrieved_docs": hyde_docs[:100]
    }



def guardrail_node(state: RAGState) -> RAGState:
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_LLM_MODEL"),
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    prompt = ChatPromptTemplate([
    (
        "system",
        "You are a helpful HR assistant. You classify whether a user's query "
        "can be answered using the internal company knowledge base.\n\n"
        "IN-SCOPE includes:\n"
        "- Policies, procedures, guidelines\n"
        "- Information extracted from internal documents\n"
        "- Tables, charts, figures, or images present in internal documents\n"
        "- Requests to retrieve or display content (text, table, image) from documents\n\n"
        "OUT-OF-SCOPE includes:\n"
        "- Chit-chat\n"
        "- Personal questions\n"
        "- Pure software tutorials unrelated to document content\n\n"
        "If in scope return 'in_scope', otherwise return 'out_of_scope'. "
        "Return ONLY one word."
    ),
    ("human", "{query}")
    ])

    result = (prompt | llm).invoke({"query": state["query"]})

    content = result.content

    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )

    label = content.strip().lower()


    if label == "out_of_scope":
        return {
            **state,
            "response": "out_of_scope"
        }
    return {
        **state,
        "response": "in_scope"
    }

def guardrail_route(state: RAGState) -> str:
    return state["response"]


# Uses Cohere's cross-encoder reranker.
# Unlike bi-encoders (which embed query and doc separately),
# a cross-encoder sees query + doc TOGETHER → more accurate relevance scoring.


def rerank_node(state: RAGState) -> RAGState:
   
   if state.get("response") == "image_retrieved":
        print("[rerank_node] Skipping rerank for image query")
        return {**state, "reranked_docs": state["retrieved_docs"]}
   
   co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
   docs = state["retrieved_docs"]

   if not docs:
       print("[rerank_node] No documents retrieved - skipping rerank")
       return {
           **state,
           "reranked_docs": []
       }

   MAX_DOCS = 50
   docs = docs[:MAX_DOCS]

   query = state["query"]

   if isinstance(query, list):
       query = " ".join(query)


   print(f"[rerank_node] Reranking {len(docs)} documents")
   rerank_response = co.rerank(
       model="rerank-english-v3.0",
       query=query,
       documents=[doc.page_content for doc in docs],
       top_n=10
   )
   

   # Map Cohere result indices back to LangChain Document objects
   reranked_docs = [docs[r.index] for r in rerank_response.results]


   print(f"[rerank_node] Top {len(reranked_docs)} chunks after reranking:")
   for i, r in enumerate(rerank_response.results):
       print(f"  Rank {i+1} | Cohere score: {r.relevance_score:.4f} | original index: {r.index}")


   return {**state, "reranked_docs": reranked_docs}



def rewrite_query_node(state: RAGState) -> RAGState:
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_LLM_MODEL"),
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Rewrite the user's query in one concise sentence to improve document retrieval."),
        ("human", "{query}")
    ])

    # Invoke the LLM to check relevance
    chain = prompt | llm
    rewritten = chain.invoke({"query": state["query"]})

    return {
        **state,
        "query": rewritten.content,
        "retries": state["retries"]+1
    }

# The decision node evaluates whether the retrieved chunks are relevant. If not,
# it will rephrase the query and send it back to the vector search node.
# If the chunks are relevant, it moves to the answer generation node.

def decision_node(state: RAGState) -> RAGState:

    if state.get("response") == "image_retrieved":
        print("[decision_node] Skipping relevance decision for image query")
        return {**state, "response": "relevant"}
    
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_LLM_MODEL"),
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    # Check relevance of retrieved chunks by sending them to the LLM
    context = "\n\n".join([doc.page_content for doc in state["reranked_docs"]])
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant that evaluates document relevance. "
                   "Please return 'relevant' if the documents are relevant to the user's query, "
                   "otherwise return 'not relevant'. Only return the word 'relevant' or 'not relevant'."),
        ("human", f"Question: {state['query']}\nContext: {context}")
    ])

    # Invoke the LLM to check relevance
    chain = prompt | llm
    response = chain.invoke({"query": state["query"], "context": context})
    print(f"LLM response: {response.content[0]['text']}")

    if state.get("response") == "hybrid_ready":
        print("[generate_answer_node] Handling HYBRID response")

        llm = ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_LLM_MODEL"),
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        structured_llm = llm.with_structured_output(AIResponse)

        context = "\n\n".join([
            f"[Source: {doc.metadata.get('source', 'unknown')} | Page: {doc.metadata.get('page', '?')}]\n{doc.page_content}"
            for doc in state["reranked_docs"]
        ])

        sql_result = state.get("sql_result", "")
        sql_query = state.get("generated_sql", "")

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a financial assistant. Answer using BOTH SQL results and document context.\n"
                "- Use SQL results for numerical insights (spending summary).\n"
                "- Use document context for policy explanations.\n"
                "- Combine both into one clear answer."
            ),
            (
                "human",
                "Question: {query}\n\n"
                "SQL Query:\n{sql}\n\n"
                "SQL Result:\n{result}\n\n"
                "Document Context:\n{context}"
            )
        ])

        chain = prompt | structured_llm

        result = chain.invoke({
            "query": state["query"],
            "sql": sql_query,
            "result": sql_result,
            "context": context
        })

        response = result.model_dump()
        response["sql_query_executed"] = sql_query

        return {**state, "response": response}

    # If the response indicates relevance, proceed to generate answer node
    if response.content[0]['text'] == 'relevant':
        print("[decision_node] Documents are relevant. Proceeding to answer generation.")
        return {**state, "response": "relevant"}  # Continue to next node

    # If the response indicates the documents are not relevant, rephrase the query
    if response.content[0]['text'] == 'not relevant':
        if state["retries"] < 3:  # Check if we have retry attempts left
            print(f"[decision_node] Documents are not relevant. Rephrasing query and retrying. Retry count: {state['retries']}")
            return {**state, "response": "retry"}  # Loop back to retrieval node
        return {**state, "response": "failed"}


# ── 4. Node 3: Generate Answer ─────────────────────────────────────────────────
# Formats the top 3 reranked chunks as context and calls Gemini LLM.
# Uses structured output to enforce the AIResponse schema.


def generate_answer_node(state: RAGState) -> RAGState:
   
   if state.get("response") == "relevant":
    image_docs = [
        doc for doc in state["reranked_docs"]
        if doc.metadata.get("chunk_type") == "image"
    ]

    if image_docs:
        print("[generate_answer_node] Returning image response")

        return {
            **state,
            "response": {
                "content_type": "image",
                "images": [
                    {
                        "image_base64": doc.metadata["image_base64"],
                        "mime_type": doc.metadata["mime_type"],
                        "page": doc.metadata["page"],
                        "source": doc.metadata["source"],
                    }
                    for doc in image_docs
                ]
            }
        }

        
   llm = ChatGoogleGenerativeAI(
       model=os.getenv("GOOGLE_LLM_MODEL"),
       google_api_key=os.getenv("GOOGLE_API_KEY")
   )
   structured_llm = llm.with_structured_output(AIResponse)


   sql_result = state.get("sql_result", "")

   sql_context = ""
   if sql_result:
        sql_context = f"""
    [SQL DATA]
    {sql_result}
    """

   doc_context = "\n\n".join([
        f"[Source: {doc.metadata.get('source', 'unknown')} | Page: {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in state["reranked_docs"]
    ])

   context = sql_context + "\n\n" + doc_context


   prompt = ChatPromptTemplate.from_messages([
       (
           "system",
           """You are a financial assistant.

Answer the user's question using:
1. SQL results (for numerical insights like spending, totals)
2. Document context (for policies, rules, explanations)

IMPORTANT:
- If SQL data is present, ALWAYS use it to answer spending-related parts, summarizing spending and etc.
- If document context is present, use it for policy-related parts
- If one source is missing, still answer using the available information (DO NOT deny completely)
- Be precise and always cite the source document and page number. If the provided context is empty due to irrelevant query, politely deny the user's query.

Formatting:
- Use short paragraphs
- Use bullet points
- Highlight important numbers
- Keep it easy to read

Only deny if BOTH SQL data and document context are missing."""
          
"Always combine:\n"
"- SQL insights (numbers)\n"
"- Document explanations (policies)\n"
       ),
       ("human", "Context:\n{context}\n\nQuestion: {query}")
   ])


   chain = prompt | structured_llm
   result = chain.invoke({"context": context, "query": state["query"]})


   print(f"[generate_answer_node] Answer generated.")
   response = result.model_dump()
   response["query"] = state.get("query")
   response["sql_query_executed"] = state.get("generated_sql")
   return {**state, "response": response}


def route(state: RAGState) -> str:
    return state["response"]

# ── 5. Build the LangGraph ─────────────────────────────────────────────────────
# Three nodes wired in a simple linear sequence.
#   vector_search → rerank → generate_answer → END


def build_rag_graph():
   graph = StateGraph(RAGState)

   
   graph.add_node("router", router_node)
   graph.add_node("nl2sql", nl2sql_node)
   graph.add_node("hybrid", hybrid_node)

   graph.add_node("guardrail", guardrail_node)
   tool_node = ToolNode(search_tools)
   graph.add_node("search_agent", search_agent_node)
   graph.add_node("tools", tool_node)
   graph.add_node("search_result", search_result_node)
   graph.add_node("rerank", rerank_node)
   graph.add_node("decision", decision_node)
   graph.add_node("generate_answer", generate_answer_node)
   graph.add_node("rewrite", rewrite_query_node)

   graph.set_entry_point("router")

   # Conditional routing: "product" → nl2sql, "document" → vector_search
   graph.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {
            "product": "nl2sql",
            "document": "guardrail",
            "hybrid": "hybrid"
        }
   )

   graph.add_edge("nl2sql", END)


   graph.add_conditional_edges(
       "guardrail",
       guardrail_route,
       {
           "in_scope": "search_agent",
           "out_of_scope": "generate_answer"
       }
   )
   graph.add_conditional_edges(
       "search_agent",
       should_continue_search,
       ["tools", END]
   )
   graph.add_edge("tools", "search_result")
   graph.add_edge("search_result", "rerank")
   graph.add_edge("rerank", "decision")

   graph.add_conditional_edges(
       "decision",
       route,
       {
           "relevant": "generate_answer",
           "retry": "rewrite",
           "failed": "generate_answer"
       }
   )
   graph.add_edge("rewrite", "search_agent")
   graph.add_edge("hybrid", "generate_answer")
   graph.add_edge("generate_answer", END)

   compiled_agent = graph.compile()
   graph_image = compiled_agent.get_graph().draw_mermaid_png()
   with open("src/api/v1/agents/reranking_workflow2.png", "wb") as f:
        f.write(graph_image)
        
   return compiled_agent

    

# Compile once at module load — reused across all requests
rag_graph = build_rag_graph()


# ── 6. Public entrypoint (called by query_service.py) ─────────────────────────
def run_vector_search_agent(query: str) -> dict:
   initial_state: RAGState = {
       "query": query,
       "hyde_query": "",
       "use_hyde": "",
       "retrieved_docs": [],
       "reranked_docs": [],
       "response": {},
       "retries": 0,
       "messages": [
           {"role": "user", "content": query}
       ],
       "route": "",
       "generated_sql": "",
       "sql_result": "",
       "hybrid_sql_response": {}
   }
   final_state = rag_graph.invoke(initial_state)
   return final_state["response"]