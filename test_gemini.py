
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("ERROR: GEMINI_API_KEY is missing.")
else:
    try:
        client = genai.Client(api_key=api_key)

        print("Checking available Gemini models...\n")

        for model in client.models.list():
            if "generateContent" in (model.supported_actions or []):
                print(model.name)

    except Exception as error:
        print("Error:", error)
