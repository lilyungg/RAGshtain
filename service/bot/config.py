import os
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = ""

POSTGRES_DSN = os.getenv(
    "POSTGRES_DSN",
    "postgresql://user:password@localhost:5432/tg_bot"
)

EXTERNAL_SERVICE_URL = os.getenv(
    "EXTERNAL_SERVICE_URL",
    "http://localhost:8001/agent"
)