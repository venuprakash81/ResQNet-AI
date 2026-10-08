
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag.query import ask_question


# --------------------------------------------------
# FASTAPI CONFIGURATION
# --------------------------------------------------

app = FastAPI(
    title="ResQNet AI API",
    description="RAG-powered AI assistant for ResQNet",
    version="1.0.0"
)


# --------------------------------------------------
# CORS CONFIGURATION
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://res-q-net-front-end-venu-prakash.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"]
)


# --------------------------------------------------
# REQUEST AND RESPONSE MODELS
# --------------------------------------------------

class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=4000
    )


class ChatResponse(BaseModel):
    answer: str


# --------------------------------------------------
# HOME ENDPOINT
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "application": "ResQNet AI",
        "status": "running",
        "endpoint": "/ask",
        "documentation": "/docs"
    }


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# --------------------------------------------------
# AI CHAT ENDPOINT
# --------------------------------------------------

@app.post("/ask", response_model=ChatResponse)
def ask_ai(request: ChatRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:
        answer = ask_question(question)

        if not answer or not answer.strip():
            raise HTTPException(
                status_code=502,
                detail="The AI returned an empty response."
            )

        return ChatResponse(answer=answer)

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except RuntimeError as error:
        print("Gemini service error:", str(error))

        raise HTTPException(
            status_code=503,
            detail=(
                "All configured Gemini models are "
                "temporarily unavailable. Please try again."
            )
        )

    except HTTPException:
        raise

    except Exception as error:
        print("Unexpected AI API error:", repr(error))

        raise HTTPException(
            status_code=500,
            detail="Unable to process your question."
        )
