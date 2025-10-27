"""
Script de pruebas para verificar el sistema de análisis de atención
"""
import cv2
import numpy as np
import asyncio
import json
from pathlib import Path
import sys

# Agregar el path del proyecto
sys.path.append(str(Path(__file__).parent))

from app.models.face_detector import FaceDetector
from app.models.tracker import StudentTracker
from app.models.attention_analyzer import AttentionAnalyzer


class SystemTester:
    """Clase para probar todos los componentes del sistema"""
    
    def __init__(self):
        self.face_detector = FaceDetector()
        self.tracker = StudentTracker()
        self.analyzer = AttentionAnalyzer()
        self.test_results = {}
        
    def print_header(self, text: str):
        """Imprime un header bonito"""
        print("\n" + "="*60)
        print(f"  {text}")
        print("="*60)
    
    def print_result(self, test_name: str, passed: bool, details: str = ""):
        """Imprime resultado de un test"""
        status = "✓ PASS" if passed else "✗ FAIL"
        color = "\033[92m" if passed else "\033[91m"
        end_color = "\033[0m"
        
        print(f"{color}{status}{end_color} - {test_name}")
        if details:
            print(f"      {details}")
        
        self.test_results[test_name] = passed
    
    def test_imports(self):
        """Verifica que todas las dependencias estén instaladas"""
        self.print_header("Test 1: Verificando Dependencias")
        
        dependencies = {
            "OpenCV": "cv2",
            "NumPy": "numpy",
            "MediaPipe": "mediapipe",
            "FastAPI": "fastapi",
            "Uvicorn": "uvicorn",
            "WebSockets": "websockets"
        }
        
        all_passed = True
        for name, module in dependencies.items():
            try:
                __import__(module)
                self.print_result(f"Importar {name}", True, f"Módulo '{module}' disponible")
            except ImportError as e:
                self.print_result(f"Importar {name}", False, f"Error: {str(e)}")
                all_passed = False
        
        return all_passed
    
    def test_face_detection(self):
        """Prueba el detector de rostros"""
        self.print_header("Test 2: Detección de Rostros")
        
        try:
            # Crear imagen de prueba con un "rostro" simple
            test_image = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Dibujar una cara simple (círculo y características)
            cv2.circle(test_image, (320, 240), 80, (255, 200, 180), -1)  # Cara
            cv2.circle(test_image, (290, 220), 10, (0, 0, 0), -1)  # Ojo izquierdo
            cv2.circle(test_image, (350, 220), 10, (0, 0, 0), -1)  # Ojo derecho
            cv2.ellipse(test_image, (320, 260), (30, 15), 0, 0, 180, (0, 0, 0), 2)  # Boca
            
            # Detectar rostros
            detections = self.face_detector.detect(test_image)
            
            # El detector puede o no detectar la cara dibujada (es muy simple)
            # Pero al menos debe ejecutarse sin errores
            self.print_result(
                "Inicialización del detector",
                True,
                f"Detecciones encontradas: {len(detections)}"
            )
            
            # Test con imagen real (captura de webcam)
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    detections = self.face_detector.detect(frame)
                    self.print_result(
                        "Detección en webcam",
                        True,
                        f"Rostros detectados: {len(detections)}"
                    )
                cap.release()
            else:
                self.print_result(
                    "Detección en webcam",
                    False,
                    "No se pudo acceder a la webcam"
                )
            
            return True
            
        except Exception as e:
            self.print_result("Detección de rostros", False, f"Error: {str(e)}")
            return False
    
    def test_tracking(self):
        """Prueba el sistema de tracking"""
        self.print_header("Test 3: Sistema de Tracking")
        
        try:
            # Crear imagen de prueba
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Simular detecciones
            detections = [
                np.array([100, 100, 200, 200]),
                np.array([300, 100, 400, 200]),
                np.array([100, 300, 200, 400])
            ]
            
            # Primera actualización
            tracked1 = self.tracker.update(test_image, detections)
            self.print_result(
                "Tracking inicial",
                len(tracked1) == len(detections),
                f"Tracks creados: {len(tracked1)}"
            )
            
            # Segunda actualización (mismas detecciones ligeramente movidas)
            detections_moved = [
                np.array([105, 105, 205, 205]),
                np.array([305, 105, 405, 205]),
                np.array([105, 305, 205, 405])
            ]
            
            tracked2 = self.tracker.update(test_image, detections_moved)
            
            # Verificar que mantiene los IDs
            ids1 = set(t['id'] for t in tracked1)
            ids2 = set(t['id'] for t in tracked2)
            
            self.print_result(
                "Consistencia de IDs",
                len(ids1.intersection(ids2)) >= 2,
                f"IDs mantenidos: {len(ids1.intersection(ids2))}/3"
            )
            
            return True
            
        except Exception as e:
            self.print_result("Sistema de tracking", False, f"Error: {str(e)}")
            return False
    
    def test_attention_analysis(self):
        """Prueba el analizador de atención"""
        self.print_header("Test 4: Análisis de Atención")
        
        try:
            # Crear imagen de prueba más realista
            test_image = np.ones((480, 640, 3), dtype=np.uint8) * 200
            
            # Simular detecciones con tracking
            detections = [
                {'id': 'student_001', 'bbox': np.array([100, 100, 250, 300])},
                {'id': 'student_002', 'bbox': np.array([350, 100, 500, 300])}
            ]
            
            # Analizar frame
            results = self.analyzer.analyze_frame(test_image, detections)
            
            self.print_result(
                "Análisis de frame",
                len(results) > 0,
                f"Estudiantes analizados: {len(results)}"
            )
            
            # Verificar estructura de resultados
            if results:
                student_id = list(results.keys())[0]
                has_current = 'current_attention' in results[student_id]
                has_avg = 'avg_attention' in results[student_id]
                
                self.print_result(
                    "Estructura de resultados",
                    has_current and has_avg,
                    f"Campos presentes: current_attention={has_current}, avg_attention={has_avg}"
                )
                
                # Analizar varios frames para generar ranking
                for _ in range(10):
                    self.analyzer.analyze_frame(test_image, detections)
                
                ranking = self.analyzer.get_final_ranking()
                
                self.print_result(
                    "Generación de ranking",
                    len(ranking) > 0,
                    f"Estudiantes en ranking: {len(ranking)}"
                )
                
                if ranking:
                    print(f"\n      Ejemplo de ranking:")
                    for i, student in enumerate(ranking[:3], 1):
                        print(f"        {i}. {student['id']}: {student['attention_percentage']:.1f}%")
            
            return True
            
        except Exception as e:
            self.print_result("Análisis de atención", False, f"Error: {str(e)}")
            return False
    
    def test_webcam_access(self):
        """Prueba acceso a la webcam"""
        self.print_header("Test 5: Acceso a Webcam")
        
        try:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                self.print_result(
                    "Abrir webcam",
                    False,
                    "No se pudo abrir la webcam. Verifica que esté conectada y no esté en uso."
                )
                return False
            
            self.print_result("Abrir webcam", True, "Webcam accesible")
            
            # Leer un frame
            ret, frame = cap.read()
            
            if ret:
                h, w = frame.shape[:2]
                self.print_result(
                    "Capturar frame",
                    True,
                    f"Resolución: {w}x{h}"
                )
            else:
                self.print_result(
                    "Capturar frame",
                    False,
                    "No se pudo leer frame de la webcam"
                )
            
            cap.release()
            return ret
            
        except Exception as e:
            self.print_result("Acceso a webcam", False, f"Error: {str(e)}")
            return False
    
    def test_performance(self):
        """Prueba el rendimiento del sistema"""
        self.print_header("Test 6: Prueba de Rendimiento")
        
        try:
            import time
            
            # Crear imagen de prueba
            test_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
            
            # Medir tiempo de detección
            start = time.time()
            iterations = 10
            
            for _ in range(iterations):
                detections = self.face_detector.detect(test_image)
                if detections:
                    tracked = self.tracker.update(test_image, detections)
                    self.analyzer.analyze_frame(test_image, tracked)
            
            elapsed = time.time() - start
            fps = iterations / elapsed
            
            self.print_result(
                "Velocidad de procesamiento",
                fps >= 5,  # Al menos 5 FPS
                f"{fps:.2f} FPS (objetivo: >5 FPS)"
            )
            
            return fps >= 5
            
        except Exception as e:
            self.print_result("Prueba de rendimiento", False, f"Error: {str(e)}")
            return False
    
    def test_websocket_connection(self):
        """Prueba la conexión WebSocket (requiere servidor corriendo)"""
        self.print_header("Test 7: Conexión WebSocket")
        
        try:
            import requests
            
            # Verificar que el servidor esté corriendo
            try:
                response = requests.get("http://localhost:8000/health", timeout=2)
                server_running = response.status_code == 200
            except:
                server_running = False
            
            if not server_running:
                self.print_result(
                    "Servidor FastAPI",
                    False,
                    "Servidor no está corriendo. Inicia el backend primero."
                )
                return False
            
            self.print_result("Servidor FastAPI", True, "Servidor accesible")
            
            # Nota: Para probar WebSocket completamente necesitaríamos un cliente async
            # Por ahora solo verificamos que el servidor esté activo
            
            return True
            
        except Exception as e:
            self.print_result("Conexión WebSocket", False, f"Error: {str(e)}")
            return False
    
    def run_all_tests(self):
        """Ejecuta todos los tests"""
        print("\n" + "="*60)
        print("  INICIANDO PRUEBAS DEL SISTEMA")
        print("  Sistema de Análisis de Atención en Clases")
        print("="*60)
        
        # Ejecutar tests
        tests = [
            self.test_imports,
            self.test_face_detection,
            self.test_tracking,
            self.test_attention_analysis,
            self.test_webcam_access,
            self.test_performance,
            self.test_websocket_connection
        ]
        
        for test in tests:
            test()
        
        # Resumen
        self.print_summary()
    
    def print_summary(self):
        """Imprime resumen de resultados"""
        self.print_header("RESUMEN DE PRUEBAS")
        
        total = len(self.test_results)
        passed = sum(1 for v in self.test_results.values() if v)
        failed = total - passed
        
        print(f"\nTotal de pruebas: {total}")
        print(f"\033[92m✓ Pasadas: {passed}\033[0m")
        print(f"\033[91m✗ Fallidas: {failed}\033[0m")
        print(f"Tasa de éxito: {(passed/total)*100:.1f}%")
        
        if failed == 0:
            print("\n\033[92m¡Todas las pruebas pasaron exitosamente!\033[0m")
            print("El sistema está listo para usar.")
        else:
            print("\n\033[91mAlgunas pruebas fallaron.\033[0m")
            print("Revisa los errores anteriores y corrige los problemas.")
        
        print("\n" + "="*60 + "\n")


def main():
    """Función principal"""
    tester = SystemTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()