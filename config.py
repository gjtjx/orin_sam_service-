from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Service configuration settings."""
    
    host: str = "0.0.0.0"
    port: int = 8000
    model_type: str = "vit_h"
    model_checkpoint: str = "sam_vit_h_4b8939.pth"
    device: str = "cuda"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
