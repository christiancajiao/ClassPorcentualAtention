"""
Configuración del sistema de penalizaciones y bonificaciones
Todos los valores son ajustables según necesidades
"""

class PenaltyConfig:
    """
    Sistema de penalizaciones basado en comportamientos negativos
    Cada estudiante inicia con 100% de atención
    """
    
    # ============================================
    # PENALIZACIONES POR FRAME (30 FPS)
    # ============================================
    
    # Desviación de mirada
    # 0.2% por frame = 6% por segundo = 360% por minuto SI es constante
    PENALTY_GAZE_DEVIATION = 0.002  # 0.2% por frame
    
    # Cabeza girada o inclinada (mirando a otro lado o hacia abajo)
    # 1% por frame = 30% por segundo
    PENALTY_HEAD_SIDE = 0.01  # 1% por frame
    
    # Estar de pie (levantarse del puesto)
    # 5% por frame = 150% por segundo (muy penalizante)
    PENALTY_STANDING = 0.05  # 5% por frame
    
    # Postura encorvada (opcional, actualmente no implementado)
    PENALTY_SLOUCHING = 0.005  # 0.5% por frame
    
    # Movimientos excesivos (opcional, actualmente no implementado)
    PENALTY_EXCESSIVE_MOVEMENT = 0.003  # 0.3% por frame
    
    # ============================================
    # BONIFICACIONES
    # ============================================
    
    # Bonus por mantener buena atención sostenida
    # +1% cada minuto de atención perfecta
    BONUS_GOOD_ATTENTION = 5.0  # 1% por minuto
    FRAMES_FOR_BONUS = 5  # 60 segundos a 30 fps
    
    # ============================================
    # UMBRALES DE DETECCIÓN
    # ============================================
    
    # Mirada
    GAZE_DEVIATION_THRESHOLD = 8  # píxeles de desviación del iris
    
    # Orientación de cabeza
    HEAD_YAW_THRESHOLD = 15   # píxeles (rotación horizontal)
    HEAD_PITCH_THRESHOLD = 15  # píxeles (inclinación vertical)
    
    # Postura de pie
    STANDING_HIP_KNEE_DISTANCE = 0.15  # distancia relativa mínima
    STANDING_VISIBILITY_THRESHOLD = 0.5  # confianza mínima de MediaPipe
    
    # ============================================
    # AJUSTES AVANZADOS
    # ============================================
    
    # FPS del sistema
    FPS = 30
    
    # Score mínimo (no puede bajar de esto)
    MIN_ATTENTION_SCORE = 0.0
    
    # Score máximo (no puede subir de esto)
    MAX_ATTENTION_SCORE = 100.0
    
    #Frames para calcular promedio histórico
    HISTORY_WINDOW = 300  # 10 segundos a 30 fps
    
    # Intervalo de logging (cada cuántos frames mostrar info)
    LOG_INTERVAL = 30  # 1 segundo
    
    # ============================================
    # CLASIFICACIÓN DE NIVELES DE ATENCIÓN
    # ============================================
    
    ATTENTION_LEVELS = {
        'excellent': {'min': 90, 'color': '#10b981', 'label': 'Excelente'},
        'high': {'min': 70, 'color': '#22c55e', 'label': 'Alta'},
        'medium': {'min': 50, 'color': '#eab308', 'label': 'Media'},
        'low': {'min': 30, 'color': '#f97316', 'label': 'Baja'},
        'very_low': {'min': 0, 'color': '#ef4444', 'label': 'Muy Baja'}
    }
    
    @classmethod
    def get_attention_level(cls, score: float) -> dict:
        """Obtiene el nivel de atención basado en el score"""
        for level_name, level_data in cls.ATTENTION_LEVELS.items():
            if score >= level_data['min']:
                return {
                    'name': level_name,
                    **level_data
                }
        return cls.ATTENTION_LEVELS['very_low']
    
    @classmethod
    def calculate_penalty_per_second(cls, penalty_per_frame: float) -> float:
        """Calcula cuánto se pierde por segundo con cierta penalización"""
        return penalty_per_frame * cls.FPS
    
    @classmethod
    def calculate_penalty_per_minute(cls, penalty_per_frame: float) -> float:
        """Calcula cuánto se pierde por minuto con cierta penalización"""
        return penalty_per_frame * cls.FPS * 60
    
    @classmethod
    def print_penalty_info(cls):
        """Imprime información sobre las penalizaciones configuradas"""
        print("\n" + "="*60)
        print("CONFIGURACIÓN DE PENALIZACIONES")
        print("="*60)
        
        print("\n📊 PENALIZACIONES:")
        print(f"  Desviación de mirada:")
        print(f"    - Por frame: {cls.PENALTY_GAZE_DEVIATION}%")
        print(f"    - Por segundo: {cls.calculate_penalty_per_second(cls.PENALTY_GAZE_DEVIATION):.2f}%")
        print(f"    - Por minuto: {cls.calculate_penalty_per_minute(cls.PENALTY_GAZE_DEVIATION):.2f}%")
        
        print(f"\n  Cabeza girada/inclinada:")
        print(f"    - Por frame: {cls.PENALTY_HEAD_SIDE}%")
        print(f"    - Por segundo: {cls.calculate_penalty_per_second(cls.PENALTY_HEAD_SIDE):.2f}%")
        print(f"    - Por minuto: {cls.calculate_penalty_per_minute(cls.PENALTY_HEAD_SIDE):.2f}%")
        
        print(f"\n  Estar de pie:")
        print(f"    - Por frame: {cls.PENALTY_STANDING}%")
        print(f"    - Por segundo: {cls.calculate_penalty_per_second(cls.PENALTY_STANDING):.2f}%")
        print(f"    - Por minuto: {cls.calculate_penalty_per_minute(cls.PENALTY_STANDING):.2f}%")
        
        print("\n✨ BONIFICACIONES:")
        print(f"  Buena atención sostenida:")
        print(f"    - Bonus: +{cls.BONUS_GOOD_ATTENTION}%")
        print(f"    - Frecuencia: cada {cls.FRAMES_FOR_BONUS/cls.FPS:.0f} segundos")
        
        print("\n🎯 UMBRALES:")
        print(f"  Desviación de mirada: {cls.GAZE_DEVIATION_THRESHOLD} píxeles")
        print(f"  Rotación cabeza (yaw): {cls.HEAD_YAW_THRESHOLD} píxeles")
        print(f"  Inclinación cabeza (pitch): {cls.HEAD_PITCH_THRESHOLD} píxeles")
        
        print("\n" + "="*60 + "\n")


# Configuración por defecto
config = PenaltyConfig()


# ============================================
# PERFILES PREDEFINIDOS
# ============================================

class StrictProfile(PenaltyConfig):
    """Perfil estricto - penalizaciones más fuertes"""
    PENALTY_GAZE_DEVIATION = 0.005  # 0.5%
    PENALTY_HEAD_SIDE = 0.02        # 2%
    PENALTY_STANDING = 0.10         # 10%
    GAZE_DEVIATION_THRESHOLD = 5
    HEAD_YAW_THRESHOLD = 10


class LenientProfile(PenaltyConfig):
    """Perfil tolerante - penalizaciones más suaves"""
    PENALTY_GAZE_DEVIATION = 0.001  # 0.1%
    PENALTY_HEAD_SIDE = 0.005       # 0.5%
    PENALTY_STANDING = 0.03         # 3%
    GAZE_DEVIATION_THRESHOLD = 12
    HEAD_YAW_THRESHOLD = 20
    BONUS_GOOD_ATTENTION = 2.0      # Bonus más generoso


class BalancedProfile(PenaltyConfig):
    """Perfil balanceado - valores por defecto"""
    pass  # Usa los valores por defecto


# ============================================
# EJEMPLO DE USO
# ============================================

if __name__ == "__main__":
    # Mostrar configuración actual
    config.print_penalty_info()
    
    # Ejemplo de cambio a perfil estricto
    print("\n🔥 PERFIL ESTRICTO:")
    print("="*60)
    strict = StrictProfile()
    strict.print_penalty_info()
    
    # Ejemplo de cambio a perfil tolerante
    print("\n😊 PERFIL TOLERANTE:")
    print("="*60)
    lenient = LenientProfile()
    lenient.print_penalty_info()