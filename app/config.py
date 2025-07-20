from pydantic_settings import BaseSettings
from pydantic import Field

from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True) 

class Settings(BaseSettings):
    openai_key: str         = Field(alias="OPENAI_API_KEY")
    telegram_token: str     = Field(alias="TELEGRAM_BOT_TOKEN")
    chroma_host: str        = "localhost"
    chroma_port: int        = 8000
    embed_model: str        = "text-embedding-3-small"

    class Config:
        env_file = ".env"   # ← this line makes Pydantic load .env for you
        extra = "ignore"

settings = Settings()       # singleton if you like