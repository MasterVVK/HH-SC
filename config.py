import os
from dotenv import load_dotenv

load_dotenv()

PROXY_URL = os.getenv("PROXY_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")