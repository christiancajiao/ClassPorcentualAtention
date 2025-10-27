"""
Estimador de Postura Corporal usando MediaPipe Pose
Analiza la postura de estudiantes para determinar nivel de atención
"""
import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Dict, Tuple
import math


class PoseEstimator:
    """
    Estima la postura corporal de una persona usando MediaPipe Pose
    """
    
    def __init__(self, 
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Inicializa el estimador de postura
        
        Args:
            min_detection_confidence: Confianza mínima para detección inicial
            min_tracking_confidence: Confianza mínima para tracking
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # 0, 1, o 2 (más complejo = más preciso pero más lento)
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Índices de landmarks clave
        self.NOSE = 0
        self.LEFT_EYE = 2
        self.RIGHT_EYE = 5
        self.LEFT_SHOULDER = 11
        self.RIGHT_SHOULDER = 12
        self.LEFT_ELBOW = 13
        self.RIGHT_ELBOW = 14
        self.LEFT_WRIST = 15
        self.RIGHT_WRIST = 16
        self.LEFT_HIP = 23
        self.RIGHT_HIP = 24
        
    def estimate(self, frame: np.ndarray, roi: Optional[np.ndarray] = None) -> Optional[Dict]:
        """
        Estima la postura en un frame o región de interés
        
        Args:
            frame: Frame completo del video
            roi: Región de interés (opcional)
            
        Returns:
            Diccionario con información de postura o None si no se detecta
        """
        # Usar ROI si se proporciona, sino usar frame completo
        image_to_process = roi if roi is not None else frame
        
        if image_to_process is None or image_to_process.size == 0:
            return None
        
        # Convertir BGR a RGB
        rgb_image = cv2.cvtColor(image_to_process, cv2.COLOR_BGR2RGB)
        
        # Procesar imagen
        results = self.pose.process(rgb_image)
        
        if not results.pose_landmarks:
            return None
        
        # Extraer landmarks
        landmarks = results.pose_landmarks.landmark
        h, w = image_to_process.shape[:2]
        
        # Calcular métricas de postura
        posture_data = {
            'landmarks': landmarks,
            'posture_score': self._calculate_posture_score(landmarks, w, h),
            'head_tilt': self._calculate_head_tilt(landmarks),
            'shoulder_alignment': self._calculate_shoulder_alignment(landmarks),
            'spine_angle': self._calculate_spine_angle(landmarks),
            'slouching': self._detect_slouching(landmarks),
            'engagement_level': self._calculate_engagement_level(landmarks, w, h)
        }
        
        return posture_data
    
    def _calculate_posture_score(self, landmarks, width: int, height: int) -> float:
        """
        Calcula un score general de postura (0-100)
        Mayor score = mejor postura = más atención
        """
        try:
            # Obtener puntos clave
            nose = landmarks[self.NOSE]
            left_shoulder = landmarks[self.LEFT_SHOULDER]
            right_shoulder = landmarks[self.RIGHT_SHOULDER]
            
            # 1. Verificar simetría de hombros (50% del score)
            shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
            symmetry_score = max(0, 100 - (shoulder_diff * 1000))
            
            # 2. Verificar alineación vertical (50% del score)
            shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
            head_shoulder_distance = abs(nose.y - shoulder_center_y)
            
            # Distancia ideal: 0.15-0.25 (normalizado)
            if 0.15 <= head_shoulder_distance <= 0.25:
                alignment_score = 100
            else:
                deviation = min(abs(head_shoulder_distance - 0.2), 0.2)
                alignment_score = max(0, 100 - (deviation * 500))
            
            # Score final
            total_score = (symmetry_score * 0.5) + (alignment_score * 0.5)
            
            return round(total_score, 2)
            
        except Exception as e:
            return 0.0
    
    def _calculate_head_tilt(self, landmarks) -> float:
        """
        Calcula el ángulo de inclinación de la cabeza en grados
        0° = cabeza recta, positivo = inclinado a la derecha, negativo = izquierda
        """
        try:
            left_eye = landmarks[self.LEFT_EYE]
            right_eye = landmarks[self.RIGHT_EYE]
            
            # Calcular ángulo
            dx = right_eye.x - left_eye.x
            dy = right_eye.y - left_eye.y
            
            angle = math.degrees(math.atan2(dy, dx))
            
            return round(angle, 2)
            
        except Exception:
            return 0.0
    
    def _calculate_shoulder_alignment(self, landmarks) -> float:
        """
        Calcula el nivel de alineación de los hombros (0-100)
        100 = perfectamente alineados
        """
        try:
            left_shoulder = landmarks[self.LEFT_SHOULDER]
            right_shoulder = landmarks[self.RIGHT_SHOULDER]
            
            # Diferencia en altura
            height_diff = abs(left_shoulder.y - right_shoulder.y)
            
            # Convertir a score (menor diferencia = mejor)
            alignment_score = max(0, 100 - (height_diff * 500))
            
            return round(alignment_score, 2)
            
        except Exception:
            return 0.0
    
    def _calculate_spine_angle(self, landmarks) -> float:
        """
        Calcula el ángulo de la columna vertebral
        90° = perfectamente recto, <90° = encorvado hacia adelante
        """
        try:
            nose = landmarks[self.NOSE]
            left_shoulder = landmarks[self.LEFT_SHOULDER]
            right_shoulder = landmarks[self.RIGHT_SHOULDER]
            left_hip = landmarks[self.LEFT_HIP]
            right_hip = landmarks[self.RIGHT_HIP]
            
            # Puntos medios
            shoulder_mid = np.array([
                (left_shoulder.x + right_shoulder.x) / 2,
                (left_shoulder.y + right_shoulder.y) / 2
            ])
            
            hip_mid = np.array([
                (left_hip.x + right_hip.x) / 2,
                (left_hip.y + right_hip.y) / 2
            ])
            
            # Vector de la columna
            spine_vector = shoulder_mid - hip_mid
            
            # Ángulo respecto a la vertical
            vertical_vector = np.array([0, -1])
            
            # Calcular ángulo
            cos_angle = np.dot(spine_vector, vertical_vector) / (
                np.linalg.norm(spine_vector) * np.linalg.norm(vertical_vector)
            )
            angle = math.degrees(math.acos(np.clip(cos_angle, -1, 1)))
            
            return round(angle, 2)
            
        except Exception:
            return 90.0
    
    def _detect_slouching(self, landmarks) -> bool:
        """
        Detecta si la persona está encorvada
        """
        try:
            spine_angle = self._calculate_spine_angle(landmarks)
            
            # Si el ángulo de la columna es menor a 75°, está encorvado
            return spine_angle < 75.0
            
        except Exception:
            return False
    
    def _calculate_engagement_level(self, landmarks, width: int, height: int) -> str:
        """
        Clasifica el nivel de engagement basado en postura
        """
        posture_score = self._calculate_posture_score(landmarks, width, height)
        slouching = self._detect_slouching(landmarks)
        
        if slouching:
            return "low"
        elif posture_score >= 70:
            return "high"
        elif posture_score >= 50:
            return "medium"
        else:
            return "low"
    
    def draw_pose(self, frame: np.ndarray, landmarks, 
                  roi_offset: Tuple[int, int] = (0, 0)) -> np.ndarray:
        """
        Dibuja los landmarks de postura en el frame
        
        Args:
            frame: Frame donde dibujar
            landmarks: Landmarks de MediaPipe
            roi_offset: Offset si se usó ROI (x, y)
            
        Returns:
            Frame con landmarks dibujados
        """
        output = frame.copy()
        h, w = frame.shape[:2]
        offset_x, offset_y = roi_offset
        
        # Dibujar conexiones
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        
        # Crear objeto de landmarks ajustado si hay offset
        if offset_x != 0 or offset_y != 0:
            # Ajustar coordenadas
            adjusted_landmarks = type('obj', (object,), {
                'landmark': [
                    type('obj', (object,), {
                        'x': lm.x + offset_x / w,
                        'y': lm.y + offset_y / h,
                        'z': lm.z,
                        'visibility': lm.visibility
                    })() for lm in landmarks
                ]
            })()
        else:
            adjusted_landmarks = type('obj', (object,), {'landmark': landmarks})()
        
        # Dibujar
        mp_drawing.draw_landmarks(
            output,
            adjusted_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=1)
        )
        
        return output
    
    def get_key_points(self, landmarks, width: int, height: int) -> Dict[str, Tuple[int, int]]:
        """
        Obtiene las coordenadas de puntos clave en píxeles
        
        Returns:
            Diccionario con puntos clave {nombre: (x, y)}
        """
        points = {}
        
        key_landmarks = {
            'nose': self.NOSE,
            'left_shoulder': self.LEFT_SHOULDER,
            'right_shoulder': self.RIGHT_SHOULDER,
            'left_elbow': self.LEFT_ELBOW,
            'right_elbow': self.RIGHT_ELBOW,
            'left_wrist': self.LEFT_WRIST,
            'right_wrist': self.RIGHT_WRIST,
            'left_hip': self.LEFT_HIP,
            'right_hip': self.RIGHT_HIP
        }
        
        for name, idx in key_landmarks.items():
            landmark = landmarks[idx]
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            points[name] = (x, y)
        
        return points
    
    def close(self):
        """
        Libera recursos
        """
        self.pose.close()


# Función auxiliar para uso rápido
def analyze_posture(frame: np.ndarray, roi: Optional[np.ndarray] = None) -> Optional[Dict]:
    """
    Función de conveniencia para analizar postura en una imagen
    
    Args:
        frame: Frame del video
        roi: Región de interés (opcional)
        
    Returns:
        Datos de postura o None
    """
    estimator = PoseEstimator()
    result = estimator.estimate(frame, roi)
    estimator.close()
    return result