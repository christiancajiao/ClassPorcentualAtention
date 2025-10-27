import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from typing import Dict, List, Tuple
import base64
from datetime import datetime
import time

# Importar configuración de penalizaciones
try:
    from app.penalties_config import PenaltyConfig
    config = PenaltyConfig()
except ImportError:
    # Valores por defecto si no se encuentra el archivo
    class PenaltyConfig:
        PENALTY_GAZE_DEVIATION = 0.002
        PENALTY_HEAD_SIDE = 0.01
        PENALTY_STANDING = 0.05
        BONUS_GOOD_ATTENTION = 5.0
        FRAMES_FOR_BONUS = 30 * 60
        GAZE_DEVIATION_THRESHOLD = 8
        HEAD_YAW_THRESHOLD = 15
        HEAD_PITCH_THRESHOLD = 15
        STANDING_HIP_KNEE_DISTANCE = 0.15
        STANDING_VISIBILITY_THRESHOLD = 0.5
        LOG_INTERVAL = 30
    config = PenaltyConfig()


class AttentionAnalyzer:
    """
    Analiza la atención de estudiantes con sistema de PENALIZACIONES
    - Cada estudiante inicia con 100% de atención
    - Las desviaciones RESTAN puntos
    - Mantener buena atención SUMA puntos (bonus de recuperación)
    """
    
    def __init__(self):
        # MediaPipe para detección facial y pose
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_pose = mp.solutions.pose
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=30,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Historial de estudiantes
        self.students_data = {}
        self.frame_count = 0
        self.fps = 30  # Frames por segundo
        
        # PENALIZACIONES POR FRAME (ajustables desde config)
        self.PENALTY_GAZE_DEVIATION = config.PENALTY_GAZE_DEVIATION
        self.PENALTY_HEAD_SIDE = config.PENALTY_HEAD_SIDE
        self.PENALTY_STANDING = config.PENALTY_STANDING
        
        # BONIFICACIONES
        self.BONUS_GOOD_ATTENTION = config.BONUS_GOOD_ATTENTION
        self.FRAMES_FOR_BONUS = config.FRAMES_FOR_BONUS
        
        # UMBRALES
        self.GAZE_DEVIATION_THRESHOLD = config.GAZE_DEVIATION_THRESHOLD
        self.HEAD_YAW_THRESHOLD = config.HEAD_YAW_THRESHOLD
        self.HEAD_PITCH_THRESHOLD = config.HEAD_PITCH_THRESHOLD
        
        print("\n" + "="*60)
        print("🎯 SISTEMA DE ATENCIÓN INICIADO")
        print("="*60)
        print(f"  Penalización mirada: {self.PENALTY_GAZE_DEVIATION}% por frame")
        print(f"  Penalización cabeza: {self.PENALTY_HEAD_SIDE}% por frame")
        print(f"  Penalización de pie: {self.PENALTY_STANDING}% por frame")
        print(f"  Bonus por minuto: +{self.BONUS_GOOD_ATTENTION}%")
        print("="*60 + "\n")
        
    def analyze_frame(self, frame: np.ndarray, detections: List[Dict]) -> Dict:
        """
        Analiza un frame completo con todas las detecciones
        """
        self.frame_count += 1
        results = {}
        
        for detection in detections:
            student_id = detection['id']
            bbox = detection['bbox']
            
            # Extraer región de interés
            x1, y1, x2, y2 = map(int, bbox)
            roi = frame[y1:y2, x1:x2]
            
            if roi.size == 0:
                continue
            
            # Inicializar estudiante si es nuevo (comienza con 100%)
            if student_id not in self.students_data:
                self.students_data[student_id] = {
                    'id': student_id,
                    'attention_score': 100.0,  # ← INICIA EN 100%
                    'score_history': deque(maxlen=300),
                    'good_attention_frames': 0,  # Contador para bonus
                    'face_image': None,
                    'total_frames': 0,
                    'first_seen': datetime.now(),
                    'last_penalties': {
                        'gaze': 0,
                        'head_orientation': 0,
                        'standing': 0
                    }
                }
            
            # Analizar comportamiento y calcular penalizaciones
            penalties = self._analyze_behavior(roi, frame, bbox)
            
            # Aplicar penalizaciones
            current_score = self.students_data[student_id]['attention_score']
            
            total_penalty = sum(penalties.values())
            new_score = max(0, current_score - total_penalty)
            
            # Verificar si merece bonus (buena atención sostenida)
            if total_penalty == 0:  # Sin penalizaciones
                self.students_data[student_id]['good_attention_frames'] += 1
                
                # Dar bonus cada minuto de buena atención
                if self.students_data[student_id]['good_attention_frames'] >= self.FRAMES_FOR_BONUS:
                    new_score = min(100, new_score + self.BONUS_GOOD_ATTENTION)
                    self.students_data[student_id]['good_attention_frames'] = 0
                    print(f"[BONUS] {student_id} recibe +1% por mantener atención")
            else:
                # Resetear contador si hay penalizaciones
                self.students_data[student_id]['good_attention_frames'] = 0
            
            # Actualizar score
            self.students_data[student_id]['attention_score'] = new_score
            self.students_data[student_id]['score_history'].append(new_score)
            self.students_data[student_id]['total_frames'] += 1
            self.students_data[student_id]['last_penalties'] = penalties
            
            # Guardar imagen facial (primera detección buena)
            if self.students_data[student_id]['face_image'] is None:
                face_img = self._extract_face(roi)
                if face_img is not None:
                    self.students_data[student_id]['face_image'] = face_img
            
            # Logging (cada 30 frames = 1 segundo)
            if self.frame_count % config.LOG_INTERVAL == 0 and total_penalty > 0:
                print(f"\n[{student_id}] Frame {self.frame_count} | Score: {new_score:.2f}%")
                if penalties['gaze'] > 0:
                    print(f"  ⚠️  Mirada desviada: -{penalties['gaze']:.3f}%")
                if penalties['head_orientation'] > 0:
                    print(f"  ⚠️  Cabeza girada: -{penalties['head_orientation']:.3f}%")
                if penalties['standing'] > 0:
                    print(f"  ⚠️  De pie: -{penalties['standing']:.3f}%")
                print(f"  ✅ Frames buenos: {self.students_data[student_id]['good_attention_frames']}/{self.FRAMES_FOR_BONUS}")
            
            scores = list(self.students_data[student_id]['score_history'])
            avg_attention = np.mean(scores) if scores else new_score

            results[student_id] = {
                'current_attention': new_score,
                'avg_attention': avg_attention,
                'penalties': penalties,
                'good_streak': self.students_data[student_id]['good_attention_frames']
            }
        
        return results
    
    def _analyze_behavior(self, roi: np.ndarray, full_frame: np.ndarray, bbox: np.ndarray) -> Dict[str, float]:
        """
        Analiza el comportamiento y retorna PENALIZACIONES
        Retorna diccionario con penalizaciones para cada categoría
        """
        penalties = {
            'gaze': 0.0,
            'head_orientation': 0.0,
            'standing': 0.0
        }
        
        # 1. Penalización por DESVIACIÓN DE MIRADA
        gaze_penalty = self._check_gaze_deviation(roi)
        if gaze_penalty > 0:
            penalties['gaze'] = self.PENALTY_GAZE_DEVIATION
        
        # 2. Penalización por CABEZA GIRADA/INCLINADA
        head_penalty = self._check_head_orientation(roi)
        if head_penalty > 0:
            penalties['head_orientation'] = self.PENALTY_HEAD_SIDE
        
        # 3. Penalización por ESTAR DE PIE
        standing_penalty = self._check_standing(roi, bbox)
        if standing_penalty > 0:
            penalties['standing'] = self.PENALTY_STANDING
        
        return penalties
    
    def _check_gaze_deviation(self, roi: np.ndarray) -> float:
        """
        Verifica si hay desviación significativa de la mirada
        Retorna 1 si hay desviación, 0 si está mirando al frente
        """
        try:
            rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_roi)
            
            if not results.multi_face_landmarks:
                return 0
            
            landmarks = results.multi_face_landmarks[0]
            h, w = roi.shape[:2]
            
            # Landmarks de iris y ojos
            LEFT_IRIS = [474, 475, 476, 477]
            RIGHT_IRIS = [469, 470, 471, 472]
            LEFT_EYE_INDICES = [33, 133, 160, 159, 158, 157, 173]
            RIGHT_EYE_INDICES = [362, 263, 387, 386, 385, 384, 398]
            
            # Calcular posición del iris vs centro del ojo
            left_iris_center = np.mean([
                [landmarks.landmark[i].x * w, landmarks.landmark[i].y * h]
                for i in LEFT_IRIS
            ], axis=0)
            
            right_iris_center = np.mean([
                [landmarks.landmark[i].x * w, landmarks.landmark[i].y * h]
                for i in RIGHT_IRIS
            ], axis=0)
            
            left_eye_center = np.mean([
                [landmarks.landmark[i].x * w, landmarks.landmark[i].y * h]
                for i in LEFT_EYE_INDICES
            ], axis=0)
            
            right_eye_center = np.mean([
                [landmarks.landmark[i].x * w, landmarks.landmark[i].y * h]
                for i in RIGHT_EYE_INDICES
            ], axis=0)
            
            # Calcular desviación
            left_deviation = np.linalg.norm(left_iris_center - left_eye_center)
            right_deviation = np.linalg.norm(right_iris_center - right_eye_center)
            avg_deviation = (left_deviation + right_deviation) / 2
            
            # Umbral: más de X píxeles = desviación significativa
            if avg_deviation > self.GAZE_DEVIATION_THRESHOLD:
                return 1  # HAY desviación
            
            return 0  # Mirando al frente
            
        except Exception as e:
            return 0
    
    def _check_head_orientation(self, roi: np.ndarray) -> float:
        """
        Verifica si la cabeza está girada o inclinada
        Retorna 1 si está girada/inclinada, 0 si está de frente
        """
        try:
            rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_roi)
            
            if not results.multi_face_landmarks:
                return 0
            
            landmarks = results.multi_face_landmarks[0]
            h, w = roi.shape[:2]
            
            # Puntos clave
            nose_tip = landmarks.landmark[1]
            left_eye = landmarks.landmark[33]
            right_eye = landmarks.landmark[263]
            chin = landmarks.landmark[152]
            
            # Calcular YAW (rotación horizontal)
            face_center_x = (left_eye.x + right_eye.x) / 2
            yaw_deviation = abs(nose_tip.x - face_center_x) * w
            
            # Calcular PITCH (inclinación vertical)
            pitch = abs(nose_tip.y - chin.y) * h
            pitch_deviation = abs(pitch - 30)  # 30 es valor neutral
            
            # Umbrales de configuración
            if yaw_deviation > self.HEAD_YAW_THRESHOLD or pitch_deviation > self.HEAD_PITCH_THRESHOLD:
                return 1  # Cabeza girada/inclinada
            
            return 0  # Cabeza de frente
            
        except Exception as e:
            return 0
    
    def _check_standing(self, roi: np.ndarray, bbox: np.ndarray) -> float:
        """
        Verifica si la persona está de pie (detectando cuerpo completo)
        Retorna 1 si está de pie, 0 si está sentado
        """
        try:
            rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_roi)
            
            if not results.pose_landmarks:
                return 0
            
            landmarks = results.pose_landmarks.landmark
            
            # Detectar si se ven las caderas y rodillas (indica de pie)
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            left_knee = landmarks[25]
            right_knee = landmarks[26]
            
            # Verificar visibilidad
            hip_visible = left_hip.visibility > 0.5 and right_hip.visibility > 0.5
            knee_visible = left_knee.visibility > 0.5 and right_knee.visibility > 0.5
            
            # Si se ven caderas Y rodillas claramente = está de pie
            if hip_visible and knee_visible:
                # Verificar que las rodillas estén significativamente abajo de las caderas
                hip_y = (left_hip.y + right_hip.y) / 2
                knee_y = (left_knee.y + right_knee.y) / 2
                
                if knee_y - hip_y > 0.15:  # Separación vertical significativa
                    return 1  # De pie
            
            return 0  # Sentado
            
        except Exception as e:
            return 0
    
    def _extract_face(self, roi: np.ndarray) -> str:
        """
        Extrae y codifica la imagen facial en base64
        """
        try:
            face_img = cv2.resize(roi, (128, 128))
            _, buffer = cv2.imencode('.jpg', face_img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{img_base64}"
        except:
            return None
    
    def get_final_ranking(self) -> List[Dict]:
        """
        Genera el ranking final de atención
        """
        ranking = []
        
        for student_id, data in self.students_data.items():
            # El score final es el último valor registrado
            final_score = data['attention_score']
            
            ranking.append({
                'id': student_id,
                'attention_percentage': round(final_score, 2),
                'face_image': data['face_image'],
                'total_frames': data['total_frames'],
                'duration_seconds': data['total_frames'] / self.fps
            })
        
        # Ordenar por atención (mayor a menor)
        ranking.sort(key=lambda x: x['attention_percentage'], reverse=True)
        
        return ranking
    
    def reset(self):
        """
        Reinicia el análisis
        """
        self.students_data = {}
        self.frame_count = 0