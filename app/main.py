import os

from fastapi import FastAPI, HTTPException, UploadFile, File

from pydantic import BaseModel

from langchain_core.messages import HumanMessage

from dotenv import load_dotenv

# Import init/getter + helpers; avoid importing the agent singleton at import-time.
from app.agent import (
    get_agent,
    init_agent,
    build_supabase_client,
    upload_user_file,
    list_user_files,
    build_llm,
)

# Add indexing helpers
from app.agent import index_user_upload

load_dotenv()


app = FastAPI(
    title="Informatica Agent API",
    description="Personal assistant for business analysts powered by LangGraph",
    version="1.0.0"
)

# Initialize heavy dependencies (docs download/index build, DB schema introspection) at startup.
# This ensures the service fails fast on boot if required resources are unavailable (Render-friendly).
@app.on_event("startup")
def _startup_init_agent():
    init_agent()


## add cors
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


## serve HTML (static serving)
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_ui():
    return FileResponse("static/index.html")


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    # Deterministic UI hint:
    # - "plain": show plain_text only
    # - "structured": show the 4-part formatted response
    render_mode: str

    # Plain response for casual queries
    plain_text: str

    # Structured response for analysis queries
    direct_answer: str
    supporting_evidence: list[str]
    business_implication: str
    recommended_next_step: list[str]

    # Debugging
    raw_model_text: str | None = None


class UploadResponse(BaseModel):
    bucket: str
    path: str
    filename: str


def _extract_text_from_agent_result(result) -> str:
    raw = result["messages"][-1].content

    # Gemini can return .content as a list of typed blocks instead of a plain string
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        return " ".join(
            block.get("text", "")
            for block in raw
            if isinstance(block, dict) and block.get("type") == "text"
        ).strip()
    return str(raw)


def _parse_structured_answer(text: str) -> dict:
    """Parse a JSON-only model answer; fall back to a safe shape if parsing fails."""
    cleaned = (text or "").strip()

    # Some models occasionally wrap JSON in code fences; be tolerant.
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    try:
        data = __import__("json").loads(cleaned)
    except Exception:
        # Fallback: preserve content but keep API contract stable.
        return {
            "direct_answer": cleaned or "Unable to produce a structured answer.",
            "supporting_evidence": [],
            "business_implication": "",
            "recommended_next_step": [],
            "raw_model_text": text,
        }

    # Validate/normalize keys and types.
    def as_str(v) -> str:
        return v if isinstance(v, str) else ("" if v is None else str(v))

    def as_str_list(v) -> list[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return [as_str(x) for x in v if as_str(x).strip()]
        # allow a single string
        if isinstance(v, str):
            return [v] if v.strip() else []
        return [as_str(v)]

    return {
        "direct_answer": as_str(data.get("direct_answer")),
        "supporting_evidence": as_str_list(data.get("supporting_evidence")),
        "business_implication": as_str(data.get("business_implication")),
        "recommended_next_step": as_str_list(data.get("recommended_next_step")),
        "raw_model_text": text,
    }


def _is_casual_query_heuristic(message: str) -> bool:
    """Cheap local heuristic for casual detection (fallback only)."""
    m = (message or "").strip().lower()
    if not m:
        return True

    # Very short + no numbers tends to be casual
    if len(m) <= 20 and not any(ch.isdigit() for ch in m):
        return True

    casual_starters = (
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "thanks",
        "thank you",
        "who are you",
        "what can you do",
        "help",
        "how are you",
    )
    if any(m == s or m.startswith(s + " ") for s in casual_starters):
        return True

    return False


def classify_query_llm(user_input: str) -> str:
    """Lightweight LLM classification step.

    Returns: "casual" or "business_analysis".
    On any unexpected output, returns "business_analysis" (safe default).
    """
    llm = build_llm()

    prompt = f"""
Classify the user query into one of two categories:

1. business_analysis -> requires data, metrics, SQL, or structured reasoning
2. casual -> general conversation, opinions, or non-analytical questions

Query: "{user_input}"

Return only one word: business_analysis or casual.
""".strip()

    result = llm.invoke(prompt)
    label = (getattr(result, "content", result) or "").strip().lower()

    if label not in {"casual", "business_analysis"}:
        return "business_analysis"

    return label


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Lightweight classification before the agent runs.
        try:
            classification = classify_query_llm(request.message)
        except Exception:
            classification = "casual" if _is_casual_query_heuristic(request.message) else "business_analysis"

        agent = get_agent()
        result = agent.invoke({
            "messages": [HumanMessage(content=request.message)]
        })

        response_text = _extract_text_from_agent_result(result)
        structured = _parse_structured_answer(response_text)

        if classification == "casual":
            plain = structured.get("direct_answer") or response_text or ""
            return ChatResponse(
                render_mode="plain",
                plain_text=plain,
                direct_answer=structured.get("direct_answer", ""),
                supporting_evidence=structured.get("supporting_evidence", []),
                business_implication=structured.get("business_implication", ""),
                recommended_next_step=structured.get("recommended_next_step", []),
                raw_model_text=structured.get("raw_model_text"),
            )

        return ChatResponse(
            render_mode="structured",
            plain_text=structured.get("direct_answer", ""),
            direct_answer=structured.get("direct_answer", ""),
            supporting_evidence=structured.get("supporting_evidence", []),
            business_implication=structured.get("business_implication", ""),
            recommended_next_step=structured.get("recommended_next_step", []),
            raw_model_text=structured.get("raw_model_text"),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    """Upload a file and store it persistently in Supabase Storage.

    This uses the configured demo Supabase project when SUPABASE_DEMO_URL/KEY are set.
    Automatically indexes text uploads into the Chroma store so they are searchable.
    """
    try:
        supabase = build_supabase_client()
        content = await file.read()
        path = upload_user_file(
            supabase,
            filename=file.filename or "upload.txt",
            content=content,
            content_type=file.content_type or "application/octet-stream",
        )

        # Index into Chroma (requires init_agent() has run at startup).
        try:
            _ = index_user_upload(supabase, path=path)
        except Exception:
            # Keep upload successful even if indexing fails; caller can retry.
            # (If you want strict behavior, we can fail the request instead.)
            pass

        return UploadResponse(
            bucket=os.environ.get("SUPABASE_UPLOADS_BUCKET", "user_uploads"),
            path=path,
            filename=file.filename or "upload.txt",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/uploads")
async def uploads_list(prefix: str = "uploads"):
    """List uploaded files in the uploads bucket."""
    try:
        supabase = build_supabase_client()
        return {"items": list_user_files(supabase, prefix=prefix)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))