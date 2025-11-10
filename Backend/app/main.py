import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 🔒 Desactiva uso de GPU (Render usa CPU)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
import json
import asyncio
from typing import Dict
import logging
import gc
import tensorflow as tf

from app.models.face_detector import FaceDetector
from app.models.tracker import StudentTracker
from app.models.attention_analyzer import AttentionAnalyzer

# ===============================
# CONFIGURACIÓN GENERAL
# ===============================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Attention Analysis API")

# CORS para permitir conexiones desde el frontend (Render o localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ Cambiar por dominio del frontend en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Estado global de las sesiones de análisis
active_sessions: Dict[str, Dict] = {}

# ===============================
# CLASE PRINCIPAL DE SESIÓN
# ===============================
class AnalysisSession:
    """Maneja una sesión de análisis de video"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.face_detector = FaceDetector(min_detection_confidence=0.6)
        self.tracker = StudentTracker()
        self.analyzer = AttentionAnalyzer()
        self.is_active = False
        self.frame_count = 0
        
    async def process_frame(self, frame_data: str) -> Dict:
        """Procesa un frame del video"""
        try:
            # Decodificar frame en base64
            frame_bytes = base64.b64decode(frame_data.split(',')[1] if ',' in frame_data else frame_data)
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                raise ValueError("No se pudo decodificar el frame")

            self.frame_count += 1
            
            # 1️⃣ Detección de rostros
            detections = self.face_detector.detect(frame)
            
            # 2️⃣ Tracking de estudiantes
            tracked_students = self.tracker.update(frame, detections)
            
            # 3️⃣ Análisis de atención
            attention_results = self.analyzer.analyze_frame(frame, tracked_students)
            
            # 4️⃣ Formatear respuesta
            response = {
                "frame_number": self.frame_count,
                "total_students": len(tracked_students),
                "students": []
            }
            
            for student in tracked_students:
                sid = student["id"]
                if sid in attention_results:
                    response["students"].append({
                        "id": sid,
                        "bbox": student["bbox"].tolist(),
                        "current_attention": attention_results[sid]["current_attention"],
                        "avg_attention": attention_results[sid]["avg_attention"]
                    })
            
            return response

        except Exception as e:
            logger.error(f"Error procesando frame: {str(e)}")
            raise

    def get_final_results(self) -> Dict:
        """Obtiene los resultados finales del análisis"""
        ranking = self.analyzer.get_final_ranking()
        return {
            "session_id": self.session_id,
            "total_frames": self.frame_count,
            "total_students": len(ranking),
            "ranking": ranking
        }

    def cleanup(self):
        """Limpia recursos y libera memoria"""
        try:
            self.face_detector.close()
        except Exception:
            pass
        try:
            self.analyzer.reset()
        except Exception:
            pass
        try:
            self.tracker.reset()
        except Exception:
            pass

        # 🔄 Limpieza de memoria TensorFlow y sistema
        tf.keras.backend.clear_session()
        gc.collect()

        logger.info(f"Recursos liberados correctamente para sesión: {self.session_id}")

# ===============================
# ENDPOINTS API
# ===============================
@app.get("/")
async def root():
    return {"message": "Attention Analysis API", "version": "1.1.0", "status": "active"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "active_sessions": len(active_sessions)}

@app.websocket("/ws/analyze/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket para análisis en tiempo real"""
    await websocket.accept()
    logger.info(f"Nueva conexión WebSocket: {session_id}")
    
    session = AnalysisSession(session_id)
    active_sessions[session_id] = session
    session.is_active = True
    
    try:
        while session.is_active:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "frame":
                result = await session.process_frame(message["data"])
                await websocket.send_json({"type": "analysis", "data": result})
            
            elif message["type"] == "end":
                final_results = session.get_final_results()
                await websocket.send_json({"type": "final_results", "data": final_results})
                session.is_active = False
                break

    except WebSocketDisconnect:
        logger.info(f"Cliente desconectado: {session_id}")
    except Exception as e:
        logger.error(f"Error en WebSocket: {str(e)}")
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        session.cleanup()
        if session_id in active_sessions:
            del active_sessions[session_id]
        logger.info(f"Sesión cerrada: {session_id}")

@app.post("/api/sessions/{session_id}/start")
async def start_session(session_id: str):
    if session_id in active_sessions:
        raise HTTPException(status_code=400, detail="Sesión ya existe")
    session = AnalysisSession(session_id)
    active_sessions[session_id] = session
    return {"session_id": session_id, "status": "started"}

@app.get("/api/sessions/{session_id}/results")
async def get_results(session_id: str):
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")
    session = active_sessions[session_id]
    return session.get_final_results()

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")
    session = active_sessions[session_id]
    session.cleanup()
    del active_sessions[session_id]
    return {"session_id": session_id, "status": "deleted"}

@app.get("/api/sessions")
async def list_sessions():
    return {"active_sessions": list(active_sessions.keys()), "count": len(active_sessions)}

# ===============================
# EJECUCIÓN LOCAL / RENDER
# ===============================
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
