import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"), max_retries=0)

for m in ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]:
    try:
        r = client.chat.completions.create(
            model=m, max_tokens=20,
            messages=[{"role": "user", "content": "word " * 2500 + "Say hi."}])
        print(m, "-> OK, tokens used:", r.usage.total_tokens)
    except Exception as e:
        print(m, "-> ERROR:", str(e)[:400])