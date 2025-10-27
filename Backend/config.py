"""
Configuración del sistema de análisis de atención
"""
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = True
    
    # CORS
    CORS_ORIGINS: list = ["http://localhost:3000", "http://localhost:5173"]
    
    # Video Processing
    TARGET_FPS: int = 10
    VIDEO_WIDTH: int = 1280
    VIDEO_HEIGHT: int = 720
    JPEG_QUALITY: int = 80
    
    # Face Detection
    MIN_FACE_DETECTION_CONFIDENCE: float = 0.5
    FACE_DETECTION_MODEL: int = 1  # 0: corta distancia, 1: larga distancia
    
    # Tracking
    IOU_THRESHOLD: float = 0.3
    MAX_TRACK_AGE: int = 30  # frames
    
    # Attention Analysis Weights
    GAZE_WEIGHT: float = 0.40
    POSTURE_WEIGHT: float = 0.30
    FACE_ORIENTATION_WEIGHT: float = 0.30
    
    # Attention Thresholds
    HIGH_ATTENTION_THRESHOLD: float = 70.0
    MEDIUM_ATTENTION_THRESHOLD: float = 50.0
    
    # Session Management
    MAX_SESSION_DURATION: int = 300  # segundos
    SESSION_CLEANUP_INTERVAL: int = 300  # segundos
    
    # Storage
    SAVE_FACE_IMAGES: bool = True
    FACE_IMAGE_SIZE: tuple = (128, 128)
    
    # Database (opcional)
    DATABASE_URL: Optional[str] = None
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Instancia global de configuración
settings = Settings()


# Constantes útiles
ATTENTION_LEVELS = {
    "high": {"min": 70, "color": "green", "label": "Alta"},
    "medium": {"min": 50, "color": "yellow", "label": "Media"},
    "low": {"min": 0, "color": "red", "label": "Baja"}
}

def get_attention_level(score: float) -> dict:
    """Obtiene el nivel de atención basado en el score"""
    if score >= settings.HIGH_ATTENTION_THRESHOLD:
        return ATTENTION_LEVELS["high"]
    elif score >= settings.MEDIUM_ATTENTION_THRESHOLD:
        return ATTENTION_LEVELS["medium"]
    else:
        return ATTENTION_LEVELS["low"]