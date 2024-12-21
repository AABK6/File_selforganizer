import os
from google import genai
from google.genai import types

api_key = os.environ.get("GENAI_API_KEY")
client = genai.Client(api_key=api_key)
model = 'gemini-2.0-flash-exp'

response = client.models.generate_content(model=model, contents="Hello")
print(response.text)


response = client.models.generate_content(
    model='gemini-2.0-flash-exp',
    contents='high',
    config=types.GenerateContentConfig(
        system_instruction='I say high, you say...',
        temperature= 0.3,
    ),
)
print(response.text)