import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from typing import Dict, List, Tuple
import base64
from datetime import datetime, timedelta
import time
import gc

try:
    from app.penalties_config import PenaltyConfig
    config = PenaltyConfig()
except ImportError:
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
        CLEANUP_INTERVAL = 600  # segundos
        INACTIVE_TIMEOUT = 300  # segundos


class AttentionAnalyzer:
    """
    Analiza la atención de estudiantes con limpieza automática y control de memoria
    """

    def __init__(self):
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

        self.students_data = {}
        self.frame_count = 0
        self.fps = 30
        self.last_cleanup = time.time()

        # Config
        self.PENALTY_GAZE_DEVIATION = config.PENALTY_GAZE_DEVIATION
        self.PENALTY_HEAD_SIDE = config.PENALTY_HEAD_SIDE
        self.PENALTY_STANDING = config.PENALTY_STANDING
        self.BONUS_GOOD_ATTENTION = config.BONUS_GOOD_ATTENTION
        self.FRAMES_FOR_BONUS = config.FRAMES_FOR_BONUS
        self.GAZE_DEVIATION_THRESHOLD = config.GAZE_DEVIATION_THRESHOLD
        self.HEAD_YAW_THRESHOLD = config.HEAD_YAW_THRESHOLD
        self.HEAD_PITCH_THRESHOLD = config.HEAD_PITCH_THRESHOLD

        print("\n=== AttentionAnalyzer iniciado con limpieza automática ===\n")

    # --------------------------------------------------------------
    # MÉTODO PRINCIPAL
    # --------------------------------------------------------------
    def analyze_frame(self, frame: np.ndarray, detections: List[Dict]) -> Dict:
        self.frame_count += 1
        results = {}

        # Limpieza periódica de memoria
        if time.time() - self.last_cleanup > config.CLEANUP_INTERVAL:
            self.cleanup_inactive_students()
            self.last_cleanup = time.time()

        for detection in detections:
            student_id = detection['id']
            bbox = detection['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            roi = frame[y1:y2, x1:x2]

            if roi.size == 0:
                continue

            student = self.students_data.get(student_id)
            if not student:
                student = {
                    'id': student_id,
                    'attention_score': 100.0,
                    'score_history': deque(maxlen=300),
                    'good_attention_frames': 0,
                    'face_image': None,
                    'total_frames': 0,
                    'first_seen': datetime.now(),
                    'last_seen': datetime.now(),
                    'last_penalties': {}
                }
                self.students_data[student_id] = student
            else:
                student['last_seen'] = datetime.now()

            penalties = self._analyze_behavior(roi, frame, bbox)
            total_penalty = sum(penalties.values())

            new_score = max(0, student['attention_score'] - total_penalty)

            # Bonus si mantiene buena atención
            if total_penalty == 0:
                student['good_attention_frames'] += 1
                if student['good_attention_frames'] >= self.FRAMES_FOR_BONUS:
                    new_score = min(100, new_score + self.BONUS_GOOD_ATTENTION)
                    student['good_attention_frames'] = 0
            else:
                student['good_attention_frames'] = 0

            student['attention_score'] = new_score
            student['score_history'].append(new_score)
            student['total_frames'] += 1
            student['last_penalties'] = penalties

            # Guardar imagen facial (solo una vez)
            if student['face_image'] is None:
                student['face_image'] = self._extract_face(roi)

            avg_attention = np.mean(student['score_history'])
            results[student_id] = {
                'current_attention': new_score,
                'avg_attention': avg_attention,
                'penalties': penalties,
                'good_streak': student['good_attention_frames']
            }

            del roi  # liberar memoria

        gc.collect()
        return results

    # --------------------------------------------------------------
    # MÉTODOS DE ANÁLISIS
    # --------------------------------------------------------------
    def _analyze_behavior(self, roi: np.ndarray, full_frame: np.ndarray, bbox: np.ndarray) -> Dict[str, float]:
        penalties = {'gaze': 0.0, 'head_orientation': 0.0, 'standing': 0.0}
        try:
            if self._check_gaze_deviation(roi):
                penalties['gaze'] = self.PENALTY_GAZE_DEVIATION
            if self._check_head_orientation(roi):
                penalties['head_orientation'] = self.PENALTY_HEAD_SIDE
            if self._check_standing(roi, bbox):
                penalties['standing'] = self.PENALTY_STANDING
        except Exception:
            pass
        return penalties

    def _check_gaze_deviation(self, roi):
        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_roi)
        del rgb_roi
        if not results.multi_face_landmarks:
            return False
        h, w = roi.shape[:2]
        landmarks = results.multi_face_landmarks[0]
        LEFT_IRIS = [474, 475, 476, 477]
        RIGHT_IRIS = [469, 470, 471, 472]
        LEFT_EYE_INDICES = [33, 133, 160, 159, 158, 157, 173]
        RIGHT_EYE_INDICES = [362, 263, 387, 386, 385, 384, 398]
        left_iris_center = np.mean([[landmarks.landmark[i].x * w, landmarks.landmark[i].y * h] for i in LEFT_IRIS], axis=0)
        right_iris_center = np.mean([[landmarks.landmark[i].x * w, landmarks.landmark[i].y * h] for i in RIGHT_IRIS], axis=0)
        left_eye_center = np.mean([[landmarks.landmark[i].x * w, landmarks.landmark[i].y * h] for i in LEFT_EYE_INDICES], axis=0)
        right_eye_center = np.mean([[landmarks.landmark[i].x * w, landmarks.landmark[i].y * h] for i in RIGHT_EYE_INDICES], axis=0)
        avg_dev = (np.linalg.norm(left_iris_center - left_eye_center) + np.linalg.norm(right_iris_center - right_eye_center)) / 2
        return avg_dev > self.GAZE_DEVIATION_THRESHOLD

    def _check_head_orientation(self, roi):
        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_roi)
        del rgb_roi
        if not results.multi_face_landmarks:
            return False
        h, w = roi.shape[:2]
        landmarks = results.multi_face_landmarks[0]
        nose_tip = landmarks.landmark[1]
        left_eye = landmarks.landmark[33]
        right_eye = landmarks.landmark[263]
        chin = landmarks.landmark[152]
        yaw_dev = abs(nose_tip.x - ((left_eye.x + right_eye.x) / 2)) * w
        pitch_dev = abs((nose_tip.y - chin.y) * h - 30)
        return yaw_dev > self.HEAD_YAW_THRESHOLD or pitch_dev > self.HEAD_PITCH_THRESHOLD

    def _check_standing(self, roi, bbox):
        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_roi)
        del rgb_roi
        if not results.pose_landmarks:
            return False
        lmk = results.pose_landmarks.landmark
        left_hip, right_hip, left_knee, right_knee = lmk[23], lmk[24], lmk[25], lmk[26]
        if left_hip.visibility > 0.5 and right_hip.visibility > 0.5 and left_knee.visibility > 0.5 and right_knee.visibility > 0.5:
            if ((left_knee.y + right_knee.y) / 2) - ((left_hip.y + right_hip.y) / 2) > 0.15:
                return True
        return False

    def _extract_face(self, roi):
        try:
            face_img = cv2.resize(roi, (96, 96))
            _, buffer = cv2.imencode('.jpg', face_img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            del face_img, buffer
            return f"data:image/jpeg;base64,{img_base64}"
        except Exception:
            return None

    # --------------------------------------------------------------
    # UTILIDADES DE LIMPIEZA Y CIERRE
    # --------------------------------------------------------------
    def cleanup_inactive_students(self):
        """Elimina estudiantes no vistos en los últimos N segundos"""
        now = datetime.now()
        to_delete = [sid for sid, s in self.students_data.items()
                     if (now - s['last_seen']).total_seconds() > config.INACTIVE_TIMEOUT]
        for sid in to_delete:
            del self.students_data[sid]
        gc.collect()
        print(f"[CLEANUP] Eliminados {len(to_delete)} estudiantes inactivos.")

    def reset(self):
        self.students_data.clear()
        gc.collect()

    def close(self):
        """Libera modelos de MediaPipe"""
        try:
            self.face_mesh.close()
            self.pose.close()
        except Exception:
            pass
        self.reset()
        print("🧹 Recursos de AttentionAnalyzer liberados correctamente.")
