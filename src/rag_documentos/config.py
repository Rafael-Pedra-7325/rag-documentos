"""Runtime configuration, read from environment variables (and an optional ``.env``)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_LLM_MODELS = {
    "anthropic": "claude-haiku-4-5",
    "openai": "gpt-4o-mini",
    "extractive": "extractive",
}
EMBEDDING_BACKENDS = ("sentence-transformers", "hashing")
VECTOR_STORES = ("chroma", "numpy")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    """All tunable knobs of the application."""

    data_dir: Path = Path("data")
    embedding_backend: str = "sentence-transformers"
    embedding_model: str = "intfloat/multilingual-e5-small"
    vector_store: str = "chroma"
    llm_provider: str = "extractive"
    llm_model: str = "extractive"
    openai_base_url: str | None = None
    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k: int = 5
    min_score: float = 0.0
    max_upload_mb: int = 25

    def __post_init__(self) -> None:
        if self.embedding_backend not in EMBEDDING_BACKENDS:
            raise ValueError(f"EMBEDDING_BACKEND must be one of {EMBEDDING_BACKENDS}")
        if self.vector_store not in VECTOR_STORES:
            raise ValueError(f"VECTOR_STORE must be one of {VECTOR_STORES}")
        if self.llm_provider not in DEFAULT_LLM_MODELS:
            raise ValueError(f"LLM_PROVIDER must be one of {tuple(DEFAULT_LLM_MODELS)}")
        if self.chunk_size < 100:
            raise ValueError("CHUNK_SIZE must be at least 100 characters")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be >= 0 and smaller than CHUNK_SIZE")
        if self.top_k < 1:
            raise ValueError("TOP_K must be at least 1")

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def numpy_dir(self) -> Path:
        return self.data_dir / "numpy"

    @property
    def registry_path(self) -> Path:
        return self.data_dir / "documents.json"

    @classmethod
    def from_env(cls, load_dotenv_file: bool = True) -> Settings:
        if load_dotenv_file:
            try:
                from dotenv import load_dotenv
            except ImportError:  # python-dotenv is optional at runtime
                pass
            else:
                load_dotenv()

        provider = os.getenv("LLM_PROVIDER", "").strip().lower()
        if not provider:
            if os.getenv("ANTHROPIC_API_KEY"):
                provider = "anthropic"
            elif os.getenv("OPENAI_API_KEY"):
                provider = "openai"
            else:
                provider = "extractive"
        if provider not in DEFAULT_LLM_MODELS:
            raise ValueError(f"LLM_PROVIDER must be one of {tuple(DEFAULT_LLM_MODELS)}")

        return cls(
            data_dir=Path(os.getenv("DATA_DIR", "data")),
            embedding_backend=os.getenv("EMBEDDING_BACKEND", "").strip() or cls.embedding_backend,
            embedding_model=os.getenv("EMBEDDING_MODEL", "").strip() or cls.embedding_model,
            vector_store=os.getenv("VECTOR_STORE", "").strip() or cls.vector_store,
            llm_provider=provider,
            llm_model=os.getenv("LLM_MODEL", "").strip() or DEFAULT_LLM_MODELS[provider],
            openai_base_url=os.getenv("OPENAI_BASE_URL", "").strip() or None,
            chunk_size=_env_int("CHUNK_SIZE", cls.chunk_size),
            chunk_overlap=_env_int("CHUNK_OVERLAP", cls.chunk_overlap),
            top_k=_env_int("TOP_K", cls.top_k),
            min_score=_env_float("MIN_SCORE", cls.min_score),
            max_upload_mb=_env_int("MAX_UPLOAD_MB", cls.max_upload_mb),
        )
