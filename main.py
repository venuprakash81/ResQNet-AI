```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag.query import ask_question


# ============================================================
# FASTAPI CONFIGURATION
# ============================================================

app = FastAPI(
    title="ResQNet AI API",
    description="RAG-powered AI assistant for ResQNet",
    version="1.0.0"
)


# ============================================================
# CORS CONFIGURATION
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        # Vercel frontend
        "https://resqnett-venuprakash.vercel.app",

        # Previous Vercel frontend domain
        "https://res-q-net-front-end-venu-prakash.vercel.app",

        # Local development
        "http://localhost:3000",
        "http://localhost:5173",
    ],

    allow_credentials=True,

    allow_methods=[
        "GET",
        "POST",
        "PUT",
        "DELETE",
        "OPTIONS",
    ],

    allow_headers=[
        "Content-Type",
        "Authorization",
    ],
)


# ============================================================
# REQUEST MODEL
# ============================================================

class ChatRequest(BaseModel):

    question: str = Field(
        ...,
        min_length=1,
        max_length=4000
    )


# ============================================================
# RESPONSE MODEL
# ============================================================

class ChatResponse(BaseModel):

    answer: str


# ============================================================
# HOME ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {
        "application": "ResQNet AI",
        "status": "running",
        "message": "ResQNet AI API is running successfully.",
        "endpoint": "/ask",
        "health": "/health",
        "documentation": "/docs"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "service": "ResQNet AI"
    }


# ============================================================
# AI CHAT ENDPOINT
# ============================================================

@app.post(
    "/ask",
    response_model=ChatResponse
)
def ask_ai(request: ChatRequest):

    # --------------------------------------------------------
    # GET QUESTION
    # --------------------------------------------------------

    question = request.question.strip()

    # --------------------------------------------------------
    # VALIDATE QUESTION
    # --------------------------------------------------------

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    # --------------------------------------------------------
    # ASK RAG AI
    # --------------------------------------------------------

    try:

        answer = ask_question(question)

        # ----------------------------------------------------
        # CHECK EMPTY RESPONSE
        # ----------------------------------------------------

        if not answer or not answer.strip():

            raise HTTPException(
                status_code=502,
                detail="The AI returned an empty response."
            )

        # ----------------------------------------------------
        # RETURN AI RESPONSE
        # ----------------------------------------------------

        return ChatResponse(
            answer=answer
        )

    # --------------------------------------------------------
    # VALUE ERROR
    # --------------------------------------------------------

    except ValueError as error:

        print(
            "Value error:",
            str(error)
        )

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    # --------------------------------------------------------
    # GEMINI / RUNTIME ERROR
    # --------------------------------------------------------

    except RuntimeError as error:

        print(
            "Gemini service error:",
            str(error)
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "All configured Gemini models are "
                "temporarily unavailable. "
                "Please try again later."
            )
        )

    # --------------------------------------------------------
    # FASTAPI HTTP ERROR
    # --------------------------------------------------------

    except HTTPException:

        raise

    # --------------------------------------------------------
    # UNKNOWN ERROR
    # --------------------------------------------------------

    except Exception as error:

        print(
            "Unexpected AI API error:",
            repr(error)
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to process your question."
        )
```
