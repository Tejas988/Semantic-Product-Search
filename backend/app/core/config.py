from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings and configuration."""
    
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Semantic Product Search API"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "AI-powered multimodal product search using CLIP and Sentence Transformers"
    
    # CORS Configuration
    BACKEND_CORS_ORIGINS: list[str] = [
        "*"  # Allow all origins for development
    ]
    
    # ML Model Configuration
    CLIP_MODEL_NAME: str = "openai/clip-vit-base-patch32"
    SENTENCE_TRANSFORMER_MODEL: str = "all-MiniLM-L6-v2"
    
    # Gemini Configuration for LLM alignment checks
    GEMINI_API_KEY: Optional[str] = ""
    
    # Search Configuration
    MAX_RECOMMENDATIONS: int = 5
    SIMILARITY_THRESHOLD: float = 0.1  # Lowered for development with mock data
    DEVELOPMENT_MODE: bool = True  # Flag for development settings
    
    # File Upload Configuration
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_IMAGE_TYPES: list[str] = ["image/jpeg", "image/png", "image/gif"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()
