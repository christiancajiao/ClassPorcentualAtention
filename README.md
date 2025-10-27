# 🎓 Sistema de Análisis de Atención en Clases

Sistema de inteligencia artificial para analizar la atención de estudiantes en tiempo real mediante el análisis de video streaming. Utiliza visión por computadora y deep learning para detectar postura, expresión facial y dirección de la mirada.

## 🌟 Características

- ✅ **Análisis en Tiempo Real**: Procesamiento de video streaming en vivo
- 👥 **Tracking Multi-Persona**: Seguimiento individual de cada estudiante
- 📊 **Métricas de Atención**: Análisis basado en:
  - Dirección de la mirada (40%)
  - Postura corporal (30%)
  - Orientación facial (30%)
- 🏆 **Ranking Automático**: Clasificación de estudiantes por nivel de atención
- 📸 **Captura de Rostros**: Registro visual de cada participante
- 🎨 **Interfaz Moderna**: Dashboard interactivo en React

## 🏗️ Arquitectura

```
┌─────────────────┐      WebSocket      ┌──────────────────┐
│   React App     │ ◄─────────────────► │   FastAPI        │
│   (Frontend)    │                      │   (Backend)      │
└─────────────────┘                      └──────────────────┘
                                                  │
                                         ┌────────┴────────┐
                                         │                 │
                                    ┌────▼─────┐    ┌─────▼──────┐
                                    │ MediaPipe│    │  OpenCV    │
                                    │  Models  │    │ Processing │
                                    └──────────┘    └────────────┘
```

## 🚀 Instalación

### Requisitos Previos

- Python 3.10+
- Node.js 16+
- npm o yarn
- Cámara web
- (Opcional) GPU NVIDIA con CUDA para mejor rendimiento

### Opción 1: Instalación Local

#### Backend (Python)

```bash
# Clonar repositorio
git clone <tu-repositorio>
cd atencion-clases/backend

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar servidor
uvicorn app.main:app --reload
```

El backend estará disponible en `http://localhost:8000`

#### Frontend (React)

```bash
cd ../frontend

# Instalar dependencias
npm install

# Ejecutar aplicación
npm start
```

El frontend estará disponible en `http://localhost:3000`

### Opción 2: Docker (Recomendado)

```bash
# Desde la raíz del proyecto
docker-compose up --build
```

Esto iniciará automáticamente:
- Backend en `http://localhost:8000`
- Frontend en `http://localhost:3000`

## 📖 Uso

### 1. Iniciar Análisis

1. Abre la aplicación en `http://localhost:3000`
2. Haz clic en **"Iniciar Análisis"**
3. Otorga permisos de acceso a la cámara
4. El sistema comenzará a detectar y analizar estudiantes automáticamente

### 2. Durante el Análisis

- Verás estadísticas en tiempo real en el panel lateral
- Cada estudiante detectado recibirá un ID único
- Los porcentajes de atención se actualizan continuamente

### 3. Finalizar y Ver Resultados

1. Haz clic en **"Detener Análisis"**
2. Se generará automáticamente un ranking completo
3. Cada estudiante mostrará:
   - Foto capturada
   - Porcentaje promedio de atención
   - Duración de participación

### 4. Nuevo Análisis

Haz clic en **"Nuevo Análisis"** para limpiar datos y comenzar de nuevo

## 🎯 API Endpoints

### WebSocket

**`WS /ws/analyze/{session_id}`**

Conexión WebSocket para streaming en tiempo real.

**Enviar frame:**
```json
{
  "type": "frame",
  "data": "base64_encoded_image"
}
```

**Finalizar sesión:**
```json
{
  "type": "end"
}
```

**Recibir análisis:**
```json
{
  "type": "analysis",
  "data": {
    "frame_number": 123,
    "total_students": 5,
    "students": [...]
  }
}
```

### REST API

**`POST /api/sessions/{session_id}/start`**
Inicia una nueva sesión de análisis

**`GET /api/sessions/{session_id}/results`**
Obtiene resultados de una sesión

**`GET /api/sessions`**
Lista sesiones activas

**`DELETE /api/sessions/{session_id}`**
Elimina una sesión

## ⚙️ Configuración

Crea un archivo `.env` en `/backend` para personalizar:

```env
# API
API_HOST=0.0.0.0
API_PORT=8000

# CORS
CORS_ORIGINS=http://localhost:3000

# Video Processing
TARGET_FPS=10
VIDEO_WIDTH=1280
VIDEO_HEIGHT=720

# Detection
MIN_FACE_DETECTION_CONFIDENCE=0.5

# Attention Weights
GAZE_WEIGHT=0.40
POSTURE_WEIGHT=0.30
FACE_ORIENTATION_WEIGHT=0.30

# Thresholds
HIGH_ATTENTION_THRESHOLD=70.0
MEDIUM_ATTENTION_THRESHOLD=50.0
```

## 🧠 Modelos de IA Utilizados

### 1. MediaPipe Face Mesh
- **Propósito**: Detección facial y análisis de mirada
- **468 landmarks** faciales en 3D
- Tracking de iris para dirección de mirada

### 2. MediaPipe Pose
- **Propósito**: Análisis de postura corporal
- 33 puntos clave del cuerpo
- Detección de inclinación y posición

### 3. Algoritmo de Tracking Personalizado
- **Método**: IoU + similitud de características
- Mantiene identidad consistente de estudiantes
- Manejo de oclusiones y movimientos

## 📊 Cálculo de Atención

El score de atención (0-100%) se calcula mediante:

```
Atención = (Mirada × 40%) + (Postura × 30%) + (Orientación × 30%)
```

**Componentes:**

1. **Mirada (40%)**: Posición del iris respecto al centro del ojo
   - Mirada al frente = 100%
   - Mirada desviada = score reducido

2. **Postura (30%)**: Inclinación y simetría corporal
   - Postura erguida = 100%
   - Postura encorvada = score reducido

3. **Orientación Facial (30%)**: Ángulo de rotación de la cara
   - Cara frontal = 100%
   - Cara girada = score reducido

## 🎨 Estructura del Proyecto

```
atencion-clases/
├── backend/
│   ├── app/
│   │   ├── main.py                 # API principal
│   │   ├── models/
│   │   │   ├── attention_analyzer.py   # Motor de análisis
│   │   │   ├── face_detector.py        # Detección facial
│   │   │   └── tracker.py              # Sistema de tracking
│   │   └── utils/
│   ├── requirements.txt
│   ├── config.py
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.jsx                # Componente principal
│   │   ├── components/
│   │   └── services/
│   ├── package.json
│   └── Dockerfile
├── docker-compose.yml
└── README.md
```

## 🔧 Troubleshooting

### Error: "No se pudo acceder a la cámara"
- Verifica permisos del navegador
- Asegúrate de usar HTTPS o localhost
- Comprueba que la cámara no esté en uso

### WebSocket se desconecta
- Verifica que el backend esté corriendo
- Revisa firewall/antivirus
- Comprueba los logs del servidor

### Bajo rendimiento
- Reduce resolución de video en config
- Disminuye TARGET_FPS
- Considera usar GPU con CUDA

### No detecta rostros
- Mejora iluminación de la sala
- Ajusta MIN_FACE_DETECTION_CONFIDENCE
- Verifica distancia de la cámara

## 🚀 Optimizaciones Futuras

- [ ] Integración con base de datos (MongoDB/PostgreSQL)
- [ ] Reconocimiento facial para identificación automática
- [ ] Análisis de emociones avanzado
- [ ] Reportes PDF exportables
- [ ] Dashboard administrativo
- [ ] API de alertas en tiempo real
- [ ] Modo multi-cámara
- [ ] Análisis histórico y tendencias

## 📝 Licencia

Este proyecto es de código abierto bajo licencia MIT.

## 👥 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📧 Contacto

Para preguntas o sugerencias, abre un issue en el repositorio.

---

**Nota**: Este sistema está diseñado para fines educativos. Asegúrate de cumplir con las regulaciones de privacidad y obtener consentimiento apropiado antes de usar en entornos reales.