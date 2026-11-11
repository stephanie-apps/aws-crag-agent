import os
from pathlib import Path

from pydantic import DirectoryPath, Field, SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict 

BASE_DIR = Path(__file__).resolve().parent.parent


class AppSettings(BaseSettings):
    api_name: str = "AWS EC2 C-RAG Agent"
    api_version: str = "v1.0.0"
    api_log_level: str = "INFO"
    allowed_hosts: list[str] = ["*"]
    allowed_origins: list[str] = ["*"]

    postgres_user: str = "postgres"
    postgres_password: SecretStr 
    postgres_db: str = "postgres"
    postgres_host: str = "postgres"
    postgres_port: int = 5432

    groq_api_key: SecretStr
    prompts_dir: DirectoryPath = Field(
        default_factory=lambda: BASE_DIR / "features" / "agent" / "v1" / "prompts"
    )

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    documents_folder: DirectoryPath = Field(
        default_factory=lambda: BASE_DIR
    )
    documents_github_user: str
    documents_github_repo: str
    documents_github_branch: str
    documents_github_token: SecretStr 
    vector_collection_name: str = "aws_ec2_documentation"
    vector_db_type: str = "pgvector"

    @computed_field
    @property
    def sync_database_url(self) -> str:
     return f"postgresql://{self.postgres_user}:{self.postgres_password.get_secret_value()}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    @computed_field
    @property
    def async_database_url(self) -> str:
     return f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password.get_secret_value()}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"


    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = AppSettings()  # type: ignore
os.environ["GROQ_API_KEY"] = settings.groq_api_key.get_secret_value()
