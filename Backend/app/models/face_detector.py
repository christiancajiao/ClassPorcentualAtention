import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple

class FaceDetector:
    """
    Detector de rostros optimizado para entornos de clase
    Usa MediaPipe Face Detection para mejor rendimiento en tiempo real
    """
    
    def __init__(self, min_detection_confidence: float = 0.5, model_selection: int = 1):
        """
        Args:
            min_detection_confidence: Confianza mínima para detección (0-1)
            model_selection: 0 para detección a corta distancia, 1 para larga distancia
        """
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence,
            model_selection=model_selection  # 1 = mejor para clases (más distancia)
        )
        
        # Fallback a Haar Cascade si MediaPipe falla
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
    def detect(self, frame: np.ndarray) -> List[np.ndarray]:
        """
        Detecta rostros en el frame
        
        Args:
            frame: Frame BGR del video
            
        Returns:
            Lista de bounding boxes en formato [x1, y1, x2, y2]
        """
        h, w = frame.shape[:2]
        
        # Convertir BGR a RGB para MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detectar rostros con MediaPipe
        results = self.face_detection.process(rgb_frame)
        
        detections = []
        
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                
                # Convertir coordenadas relativas a absolutas
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = int((bbox.xmin + bbox.width) * w)
                y2 = int((bbox.ymin + bbox.height) * h)
                
                # Asegurar que estén dentro de los límites
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                detections.append(np.array([x1, y1, x2, y2]))
        
        # Si MediaPipe no detecta nada, usar Haar Cascade como fallback
        if len(detections) == 0:
            detections = self._detect_haar(frame)
        
        return detections
    
    def _detect_haar(self, frame: np.ndarray) -> List[np.ndarray]:
        """
        Detección usando Haar Cascade (fallback)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        detections = []
        for (x, y, w, h) in faces:
            detections.append(np.array([x, y, x + w, y + h]))
        
        return detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[np.ndarray], 
                       labels: List[str] = None) -> np.ndarray:
        """
        Dibuja las detecciones en el frame (útil para debugging)
        
        Args:
            frame: Frame donde dibujar
            detections: Lista de bounding boxes
            labels: Etiquetas opcionales para cada detección
            
        Returns:
            Frame con detecciones dibujadas
        """
        output = frame.copy()
        
        for i, bbox in enumerate(detections):
            x1, y1, x2, y2 = map(int, bbox)
            
            # Dibujar rectángulo
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Dibujar etiqueta si existe
            if labels and i < len(labels):
                label = labels[i]
                cv2.putText(output, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return output
    
    def close(self):
        """
        Libera recursos
        """
        self.face_detection.close()