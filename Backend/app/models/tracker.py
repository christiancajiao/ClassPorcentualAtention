import cv2
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
import uuid

class StudentTracker:
    """
    Sistema de tracking para seguir a estudiantes individuales a través del video
    Usa algoritmo de tracking simple basado en IoU y características faciales
    """
    
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30):
        self.tracked_students = {}  # {student_id: tracking_info}
        self.next_id = 0
        self.iou_threshold = iou_threshold
        self.max_age = max_age  # Frames máximos sin detección antes de eliminar
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
    def update(self, frame: np.ndarray, detections: List[np.ndarray]) -> List[Dict]:
        """
        Actualiza el tracking con nuevas detecciones
        
        Args:
            frame: Frame actual del video
            detections: Lista de bounding boxes [x1, y1, x2, y2]
            
        Returns:
            Lista de detecciones con IDs asignados
        """
        # Incrementar edad de todos los tracks existentes
        for student_id in list(self.tracked_students.keys()):
            self.tracked_students[student_id]['age'] += 1
            
            # Eliminar tracks muy antiguos
            if self.tracked_students[student_id]['age'] > self.max_age:
                del self.tracked_students[student_id]
        
        if len(detections) == 0:
            return []
        
        # Si no hay tracks previos, crear nuevos
        if len(self.tracked_students) == 0:
            return self._initialize_tracks(frame, detections)
        
        # Asociar detecciones con tracks existentes
        matched_tracks = self._associate_detections(frame, detections)
        
        return matched_tracks
    
    def _initialize_tracks(self, frame: np.ndarray, detections: List[np.ndarray]) -> List[Dict]:
        """
        Inicializa tracks para las primeras detecciones
        """
        results = []
        
        for bbox in detections:
            student_id = self._generate_id()
            x1, y1, x2, y2 = map(int, bbox)
            
            # Extraer características de la cara
            face_features = self._extract_face_features(frame, bbox)
            
            self.tracked_students[student_id] = {
                'bbox': bbox,
                'features': face_features,
                'age': 0,
                'confidence': 1.0
            }
            
            results.append({
                'id': student_id,
                'bbox': bbox,
                'confidence': 1.0
            })
        
        return results
    
    def _associate_detections(self, frame: np.ndarray, detections: List[np.ndarray]) -> List[Dict]:
        """
        Asocia detecciones actuales con tracks existentes usando IoU y características
        """
        if len(detections) == 0:
            return []
        
        # Calcular matriz de costos
        cost_matrix = np.zeros((len(self.tracked_students), len(detections)))
        track_ids = list(self.tracked_students.keys())
        
        for i, student_id in enumerate(track_ids):
            prev_bbox = self.tracked_students[student_id]['bbox']
            prev_features = self.tracked_students[student_id]['features']
            
            for j, curr_bbox in enumerate(detections):
                # Calcular IoU
                iou = self._calculate_iou(prev_bbox, curr_bbox)
                
                # Calcular similitud de características
                curr_features = self._extract_face_features(frame, curr_bbox)
                feature_similarity = self._compare_features(prev_features, curr_features)
                
                # Costo combinado (menor es mejor)
                # Invertimos IoU porque mayor IoU = menor costo
                cost = (1 - iou) * 0.6 + (1 - feature_similarity) * 0.4
                cost_matrix[i, j] = cost
        
        # Asignación húngara simple (greedy)
        matched_tracks = []
        used_detections = set()
        used_tracks = set()
        
        # Ordenar por costo
        matches = []
        for i in range(len(track_ids)):
            for j in range(len(detections)):
                matches.append((cost_matrix[i, j], i, j))
        matches.sort()
        
        # Asignar matches
        for cost, i, j in matches:
            if i in used_tracks or j in used_detections:
                continue
            if cost < 0.7:  # Umbral de costo
                student_id = track_ids[i]
                bbox = detections[j]
                
                # Actualizar track
                self.tracked_students[student_id]['bbox'] = bbox
                self.tracked_students[student_id]['features'] = self._extract_face_features(frame, bbox)
                self.tracked_students[student_id]['age'] = 0
                
                matched_tracks.append({
                    'id': student_id,
                    'bbox': bbox,
                    'confidence': 1 - cost
                })
                
                used_tracks.add(i)
                used_detections.add(j)
        
        # Crear nuevos tracks para detecciones no asociadas
        for j, bbox in enumerate(detections):
            if j not in used_detections:
                student_id = self._generate_id()
                face_features = self._extract_face_features(frame, bbox)
                
                self.tracked_students[student_id] = {
                    'bbox': bbox,
                    'features': face_features,
                    'age': 0,
                    'confidence': 1.0
                }
                
                matched_tracks.append({
                    'id': student_id,
                    'bbox': bbox,
                    'confidence': 1.0
                })
        
        return matched_tracks
    
    def _calculate_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """
        Calcula Intersection over Union entre dos bounding boxes
        """
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Calcular intersección
        xi_min = max(x1_min, x2_min)
        yi_min = max(y1_min, y2_min)
        xi_max = min(x1_max, x2_max)
        yi_max = min(y1_max, y2_max)
        
        if xi_max < xi_min or yi_max < yi_min:
            return 0.0
        
        intersection = (xi_max - xi_min) * (yi_max - yi_min)
        
        # Calcular unión
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _extract_face_features(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """
        Extrae características simples de la cara para tracking
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Asegurar que las coordenadas estén dentro del frame
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return np.zeros(128)
        
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return np.zeros(128)
        
        try:
            # Redimensionar a tamaño fijo
            resized = cv2.resize(roi, (32, 32))
            
            # Convertir a escala de grises
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            
            # Calcular histograma como características
            hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
            hist = hist.flatten()
            
            # Normalizar
            hist = hist / (hist.sum() + 1e-7)
            
            # Extraer características adicionales (color promedio)
            color_features = cv2.mean(resized)[:3]
            color_features = np.array(color_features) / 255.0
            
            # Combinar características
            features = np.concatenate([hist, color_features])
            
            # Pad hasta 128 dimensiones
            if len(features) < 128:
                features = np.pad(features, (0, 128 - len(features)))
            else:
                features = features[:128]
            
            return features
        except:
            return np.zeros(128)
    
    def _compare_features(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """
        Compara dos vectores de características usando similitud coseno
        """
        if features1 is None or features2 is None:
            return 0.0
        
        # Similitud coseno
        dot_product = np.dot(features1, features2)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        
        # Normalizar a [0, 1]
        return (similarity + 1) / 2
    
    def _generate_id(self) -> str:
        """
        Genera un ID único para un estudiante
        """
        student_id = f"student_{self.next_id:03d}"
        self.next_id += 1
        return student_id
    
    def reset(self):
        """
        Reinicia el tracker
        """
        self.tracked_students = {}
        self.next_id = 0