import os

from fastapi import FastAPI, HTTPException

from pydantic import BaseModel

from langchain_core.messages import HumanMessage

from dotenv import load_dotenv

from app.agent import agent

load_dotenv()


app = FastAPI(
    title="Informatica Agent API",
    description="Personal assistant for business analysts powered by LangGraph",
    version="1.0.0"
)

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
    response: str


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        result = agent.invoke({
            "messages": [HumanMessage(content=request.message)]
        })

        raw = result["messages"][-1].content

        # Gemini can return .content as a list of typed blocks instead of a plain string
        if isinstance(raw, str):
            response_text = raw
        elif isinstance(raw, list):
            response_text = " ".join(
                block["text"] for block in raw
                if isinstance(block, dict) and block.get("type") == "text"
            )
        else:
            response_text = str(raw)

        return ChatResponse(response=response_text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))