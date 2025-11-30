import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

    DEFAULT_LLM_PROVIDER: str = os.getenv("DEFAULT_LLM_PROVIDER", "openai")  # "openai" or "anthropic"
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")

    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.75"))

    MAX_CITATIONS_TO_CHECK: int = int(os.getenv("MAX_CITATIONS_TO_CHECK", "50"))
    MAX_CONCURRENT_REQUESTS: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "2"))

    OPENAI_RATE_LIMIT: float = float(os.getenv("OPENAI_RATE_LIMIT", "3.0"))
    SEMANTIC_SCHOLAR_RATE_LIMIT: float = float(os.getenv("SEMANTIC_SCHOLAR_RATE_LIMIT", "1.0"))

    WEB_REQUEST_TIMEOUT: int = int(os.getenv("WEB_REQUEST_TIMEOUT", "30"))
    BROWSER_USE_TIMEOUT: int = int(os.getenv("BROWSER_USE_TIMEOUT", "60"))

    BROWSER_HEADLESS: bool = os.getenv("BROWSER_HEADLESS", "true").lower() == "true"

    PROJECT_ROOT: Path = Path(__file__).parent
    OUTPUT_DIR: Path = PROJECT_ROOT / os.getenv("OUTPUT_DIR", "output")
    DATA_DIR: Path = PROJECT_ROOT / os.getenv("DATA_DIR", "data")

    OUTPUT_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    VERBOSE: bool = os.getenv("VERBOSE", "false").lower() == "true"

    @classmethod
    def validate(cls) -> None:
        """Validate that required configuration is present."""
        errors = []

        if not cls.OPENAI_API_KEY and cls.DEFAULT_LLM_PROVIDER == "openai":
            errors.append("OPENAI_API_KEY is required when using OpenAI as the LLM provider")

        if not cls.ANTHROPIC_API_KEY and cls.DEFAULT_LLM_PROVIDER == "anthropic":
            errors.append("ANTHROPIC_API_KEY is required when using Anthropic as the LLM provider")

        if not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is required for embeddings")

        if errors:
            raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))

    @classmethod
    def get_llm_config(cls) -> dict:
        """Get LLM configuration based on the selected provider."""
        if cls.DEFAULT_LLM_PROVIDER == "openai":
            return {
                "provider": "openai",
                "model": cls.OPENAI_MODEL,
                "api_key": cls.OPENAI_API_KEY,
            }
        elif cls.DEFAULT_LLM_PROVIDER == "anthropic":
            return {
                "provider": "anthropic",
                "model": cls.ANTHROPIC_MODEL,
                "api_key": cls.ANTHROPIC_API_KEY,
            }
        else:
            raise ValueError(f"Unknown LLM provider: {cls.DEFAULT_LLM_PROVIDER}")


try:
    Config.validate()
except ValueError as e:
    print(f"Warning: {e}")
    print("Some features may not work without proper API keys configured.")
