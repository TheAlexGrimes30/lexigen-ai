from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv()

class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    DB_NAME: str
    DB_USER: str
    DB_PASS: str
    DB_HOST: str
    DB_PORT: int

    QDRANT_HOST: str
    QDRANT_PORT: int = 6333

    JWT_SECRET: str = "change_me_very_secret_key"
    JWT_EXPIRES_MINUTES: int = 60 * 24
    ADMIN_EMAIL: str = "admin@lexigen.local"

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def DATABASE_URL(self) -> str:
        """Return the PostgreSQL async connection URL."""

        return (
            f"postgresql+asyncpg://{self.DB_USER}:"
            f"{self.DB_PASS}@{self.DB_HOST}:"
            f"{self.DB_PORT}/{self.DB_NAME}"
        )


settings = Settings()