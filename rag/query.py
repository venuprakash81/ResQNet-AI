
import os
import time
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "chroma_db"

load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. Check your .env file."
    )

client = genai.Client(api_key=API_KEY)

db = chromadb.PersistentClient(path=str(DB_PATH))
collection = db.get_or_create_collection(
    name="resqnet_docs"
)

# Try models in order. Remove models that your account
# reports as unavailable.
MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
]

MAX_RETRIES = 2
RETRY_DELAY = 3


# --------------------------------------------------
# RETRIEVE PROJECT INFORMATION
# --------------------------------------------------

def retrieve_context(question, n_results=5):
    total_docs = collection.count()

    if total_docs == 0:
        return ""

    results = collection.query(
        query_texts=[question],
        n_results=min(n_results, total_docs),
        include=["documents", "metadatas", "distances"]
    )

    documents = results.get("documents", [[]])
    metadatas = results.get("metadatas", [[]])

    if not documents or not documents[0]:
        return ""

    context_parts = []

    for index, document in enumerate(documents[0]):
        metadata = (
            metadatas[0][index]
            if metadatas and metadatas[0]
            else {}
        )

        source = metadata.get("source", "Unknown source")

        context_parts.append(
            f"Source: {source}\n"
            f"Content:\n{document}"
        )

    return "\n\n---\n\n".join(context_parts)


# --------------------------------------------------
# GEMINI MODEL FALLBACK
# --------------------------------------------------

def generate_answer(prompt):
    errors = []

    for model in MODELS:
        for attempt in range(MAX_RETRIES):
            try:
                print(
                    f"Trying model: {model} "
                    f"(attempt {attempt + 1}/{MAX_RETRIES})"
                )

                response = client.models.generate_content(
                    model=model,
                    contents=prompt
                )

                answer = response.text

                if answer and answer.strip():
                    print(f"Success with model: {model}")
                    return answer.strip()

                errors.append(
                    f"{model}: Empty response"
                )
                break

            except Exception as error:
                error_message = str(error)
                print(
                    f"{model} failed: {error_message}"
                )

                errors.append(
                    f"{model}: {error_message}"
                )

                # Retry only likely temporary failures.
                temporary = any(
                    code in error_message
                    for code in [
                        "503",
                        "UNAVAILABLE",
                        "429",
                        "RESOURCE_EXHAUSTED",
                        "500",
                        "INTERNAL"
                    ]
                )

                if temporary and attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue

                break

    details = "\n".join(errors[-len(MODELS) * MAX_RETRIES:])

    raise RuntimeError(
        "All available Gemini model attempts failed.\n"
        + details
    )


# --------------------------------------------------
# MAIN QUESTION FUNCTION
# --------------------------------------------------

def ask_question(question):
    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    context = retrieve_context(question)

    if context:
        prompt = f"""
You are ResQNet AI, an assistant for the ResQNet
emergency healthcare and rescue network project.

Answer the user's question using the supplied project
information whenever it is relevant.

Rules:
1. Use the retrieved code and documentation as evidence.
2. Do not invent API endpoints, filenames, database
   fields, or implementation details.
3. If information is missing, clearly say that it
   could not be found in the indexed project files.
4. Distinguish example data from actual project data.
5. Explain technical answers clearly and practically.
6. For emergency situations, advise users to contact
   official emergency services directly. Do not claim
   that you have dispatched help.

Retrieved project information:
{context}

User question:
{question}

Answer:
"""
    else:
        prompt = f"""
You are ResQNet AI, an assistant for the ResQNet
project.

No relevant project information was found in the
indexed files for this question.

Do not invent project-specific implementation details.
Explain that the project files do not provide enough
information to answer accurately.

User question:
{question}

Answer:
"""

    return generate_answer(prompt)


# --------------------------------------------------
# COMMAND-LINE TEST
# --------------------------------------------------

if __name__ == "__main__":
    print("====================================")
    print("       ResQNet AI Assistant")
    print("====================================")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            question = input("You: ").strip()

            if question.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            if not question:
                continue

            answer = ask_question(question)
            print("\nResQNet AI:\n")
            print(answer)
            print()

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

        except Exception as error:
            print(f"\nResQNet AI error: {error}\n")
