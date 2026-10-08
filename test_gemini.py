
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

try:
    print("Checking available Gemini models...\n")

    for model in client.models.list():
        if "generateContent" in (model.supported_actions or []):
            print(model.name)

except Exception as error:
    print("Error:", error)
