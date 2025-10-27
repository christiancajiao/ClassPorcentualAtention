"""
Script para probar y visualizar el sistema de penalizaciones en tiempo real
Muestra el porcentaje de atención mientras capturas video de tu webcam

UBICACIÓN: Backend/test_penalties.py
"""
import cv2
import numpy as np
import sys
from pathlib import Path

# Agregar path del proyecto para importaciones
sys.path.append(str(Path(__file__).parent))

# Importar módulos desde app
from app.models.attention_analyzer import AttentionAnalyzer
from app.models.face_detector import FaceDetector
from app.models.tracker import StudentTracker
from app.penalties_config import PenaltyConfig, StrictProfile, LenientProfile, BalancedProfile

class AttentionTester:
    """Clase para probar el sistema de atención en vivo"""
    
    def __init__(self, config=None):
        self.config = config or PenaltyConfig()
        self.face_detector = FaceDetector()
        self.tracker = StudentTracker()
        self.analyzer = AttentionAnalyzer()
        
        # Mostrar configuración
        self.config.print_penalty_info()
        
    def run_live_test(self):
        """Ejecuta prueba en vivo con webcam"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: No se pudo abrir la webcam")
            return
        
        print("\n🎥 Iniciando prueba en vivo...")
        print("📋 Instrucciones:")
        print("  - Siéntate frente a la cámara mirando al frente")
        print("  - Tu atención inicia en 100%")
        print("  - Prueba desviar la mirada, girar la cabeza, levantarte")
        print("  - Presiona 'q' para salir")
        print("\n" + "="*60 + "\n")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detectar rostros
            detections = self.face_detector.detect(frame)
            
            # Tracking
            tracked = self.tracker.update(frame, detections)
            
            # Analizar atención
            results = self.analyzer.analyze_frame(frame, tracked)
            
            # Visualizar
            display_frame = self._draw_visualization(frame, tracked, results)
            
            cv2.imshow('Test de Atención - Presiona Q para salir', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Mostrar resultados finales
        self._show_final_results()
    
    def _draw_visualization(self, frame, tracked, results):
        """Dibuja la visualización en el frame"""
        output = frame.copy()
        h, w = output.shape[:2]
        
        for student in tracked:
            student_id = student['id']
            bbox = student['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            if student_id not in results:
                continue
            
            attention = results[student_id]['current_attention']
            penalties = results[student_id]['penalties']
            
            # Color según nivel de atención
            if attention >= 90:
                color = (0, 255, 0)  # Verde
                status = "EXCELENTE"
            elif attention >= 70:
                color = (0, 255, 255)  # Amarillo-verde
                status = "ALTA"
            elif attention >= 50:
                color = (0, 165, 255)  # Naranja
                status = "MEDIA"
            else:
                color = (0, 0, 255)  # Rojo
                status = "BAJA"
            
            # Dibujar bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)
            
            # Dibujar fondo para texto
            cv2.rectangle(output, (x1, y1 - 80), (x2, y1), color, -1)
            
            # Dibujar información
            cv2.putText(output, f"{student_id}", 
                       (x1 + 5, y1 - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            cv2.putText(output, f"Atencion: {attention:.1f}%", 
                       (x1 + 5, y1 - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            cv2.putText(output, f"Estado: {status}", 
                       (x1 + 5, y1 - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Mostrar penalizaciones activas
            penalty_y = y2 + 25
            if penalties['gaze'] > 0:
                cv2.putText(output, "⚠ Mirada desviada", 
                           (x1, penalty_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                penalty_y += 20
            
            if penalties['head_orientation'] > 0:
                cv2.putText(output, "⚠ Cabeza girada", 
                           (x1, penalty_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                penalty_y += 20
            
            if penalties['standing'] > 0:
                cv2.putText(output, "⚠ De pie", 
                           (x1, penalty_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Panel de información general
        info_panel = np.zeros((150, w, 3), dtype=np.uint8)
        cv2.putText(info_panel, "SISTEMA DE ANALISIS DE ATENCION", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.putText(info_panel, f"Estudiantes detectados: {len(tracked)}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.putText(info_panel, "PENALIZACIONES:", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
        
        cv2.putText(info_panel, f"Mirada: {self.config.PENALTY_GAZE_DEVIATION}%", 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.putText(info_panel, f"Cabeza: {self.config.PENALTY_HEAD_SIDE}%", 
                   (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.putText(info_panel, f"De pie: {self.config.PENALTY_STANDING}%", 
                   (200, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Combinar frame con panel
        output = np.vstack([info_panel, output])
        
        return output
    
    def _show_final_results(self):
        """Muestra resultados finales al terminar"""
        print("\n" + "="*60)
        print("📊 RESULTADOS FINALES")
        print("="*60)
        
        ranking = self.analyzer.get_final_ranking()
        
        if not ranking:
            print("No se detectaron estudiantes")
            return
        
        for i, student in enumerate(ranking, 1):
            print(f"\n{i}. {student['id']}")
            print(f"   Atención final: {student['attention_percentage']:.2f}%")
            print(f"   Duración: {student['duration_seconds']:.1f} segundos")
            
            level = self.config.get_attention_level(student['attention_percentage'])
            print(f"   Nivel: {level['label']}")
        
        print("\n" + "="*60 + "\n")


def main():
    """Función principal"""
    print("\n" + "="*60)
    print("🎓 SISTEMA DE ANÁLISIS DE ATENCIÓN - TEST")
    print("="*60)
    
    print("\nSelecciona un perfil de penalización:")
    print("1. Balanceado (por defecto)")
    print("2. Estricto (penalizaciones más fuertes)")
    print("3. Tolerante (penalizaciones más suaves)")
    print("4. Personalizado")
    
    choice = input("\nOpción (1-4): ").strip() or "1"
    
    if choice == "2":
        config = StrictProfile()
        print("\n✅ Usando perfil ESTRICTO")
    elif choice == "3":
        config = LenientProfile()
        print("\n✅ Usando perfil TOLERANTE")
    elif choice == "4":
        print("\n⚙️  Configuración personalizada:")
        gaze = float(input("  Penalización mirada (0.001-0.01): ") or "0.002")
        head = float(input("  Penalización cabeza (0.005-0.05): ") or "0.01")
        standing = float(input("  Penalización de pie (0.01-0.2): ") or "0.05")
        
        config = PenaltyConfig()
        config.PENALTY_GAZE_DEVIATION = gaze
        config.PENALTY_HEAD_SIDE = head
        config.PENALTY_STANDING = standing
        print("\n✅ Usando configuración personalizada")
    else:
        config = PenaltyConfig()
        print("\n✅ Usando perfil BALANCEADO")
    
    # Iniciar test
    tester = AttentionTester(config)
    tester.run_live_test()


if __name__ == "__main__":
    main()