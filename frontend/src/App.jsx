import React, { useState, useRef, useEffect } from "react";
import { Camera, Square, Users, TrendingUp, Award } from "lucide-react";

export default function AttentionAnalysisApp() {
  const [isStreaming, setIsStreaming] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [currentStats, setCurrentStats] = useState({ total: 0, students: [] });
  const [finalResults, setFinalResults] = useState(null);
  const [error, setError] = useState(null);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const streamRef = useRef(null);
  const intervalRef = useRef(null);

  // ⚠️ IMPORTANTE: Cambia esta URL por la exacta de tu backend
  const BACKEND_URL = "https://atencion-backend-n1eu.onrender.com";
  const WS_URL = BACKEND_URL.replace("https://", "wss://");

  const generateSessionId = () => {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  };

  const startStream = async () => {
    try {
      setError(null);

      console.log("🎥 Iniciando stream...");
      console.log("🔗 Backend URL:", BACKEND_URL);

      // Solicitar acceso a la cámara
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720 },
      });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      // Crear sesión
      const newSessionId = generateSessionId();
      setSessionId(newSessionId);

      // Conectar WebSocket con la URL correcta
      const websocketUrl = `${WS_URL}/ws/analyze/${newSessionId}`;
      console.log("🔌 Conectando WebSocket a:", websocketUrl);

      const ws = new WebSocket(websocketUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log("✅ WebSocket conectado exitosamente");
        setIsStreaming(true);
        setError(null);

        // Comenzar a enviar frames cada 100ms (10 fps)
        intervalRef.current = setInterval(() => {
          captureAndSendFrame();
        }, 100);
      };

      ws.onmessage = (event) => {
        console.log("📩 Mensaje recibido:", event.data);
        const message = JSON.parse(event.data);

        if (message.type === "analysis") {
          setCurrentStats({
            total: message.data.total_students,
            students: message.data.students,
          });
        } else if (message.type === "final_results") {
          setFinalResults(message.data);
          stopStream();
        } else if (message.type === "error") {
          console.error("❌ Error del servidor:", message.message);
          setError(message.message);
        }
      };

      ws.onerror = (error) => {
        console.error("❌ WebSocket error:", error);
        setError(
          "Error de conexión con el servidor. El backend puede estar durmiendo. Espera 1 minuto e intenta de nuevo."
        );
      };

      ws.onclose = (event) => {
        console.log("🔌 WebSocket cerrado:", event.code, event.reason);
        if (event.code !== 1000) {
          setError("Conexión cerrada inesperadamente. Código: " + event.code);
        }
      };
    } catch (err) {
      console.error("❌ Error iniciando stream:", err);
      setError("No se pudo acceder a la cámara. Verifica los permisos.");
    }
  };

  const captureAndSendFrame = () => {
    if (!videoRef.current || !canvasRef.current || !wsRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext("2d");

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(
      (blob) => {
        if (!blob) return;

        const reader = new FileReader();
        reader.onloadend = () => {
          if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(
              JSON.stringify({
                type: "frame",
                data: reader.result,
              })
            );
          }
        };
        reader.readAsDataURL(blob);
      },
      "image/jpeg",
      0.8
    );
  };

  const stopStream = () => {
    // Detener envío de frames
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    // Enviar señal de fin si WebSocket está abierto
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "end" }));
    }

    // Detener stream de video
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setIsStreaming(false);
  };

  const resetAnalysis = () => {
    setFinalResults(null);
    setCurrentStats({ total: 0, students: [] });
    setSessionId(null);
    setError(null);
  };

  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      if (wsRef.current) wsRef.current.close();
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">
            Sistema de Análisis de Atención
          </h1>
          <p className="text-gray-600">
            Monitoreo en tiempo real de la atención en el aula
          </p>
        </div>

        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
            {error}
          </div>
        )}

        {/* Vista de Streaming o Resultados */}
        {!finalResults ? (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Video Preview */}
            <div className="lg:col-span-2">
              <div className="bg-white rounded-lg shadow-lg p-6">
                <div
                  className="relative bg-gray-900 rounded-lg overflow-hidden"
                  style={{ aspectRatio: "16/9" }}
                >
                  <video
                    ref={videoRef}
                    className="w-full h-full object-cover"
                    playsInline
                  />
                  {!isStreaming && (
                    <div className="absolute inset-0 flex items-center justify-center">
                      <Camera className="w-24 h-24 text-gray-600" />
                    </div>
                  )}
                </div>

                <canvas ref={canvasRef} className="hidden" />

                <div className="mt-4 flex justify-center gap-4">
                  {!isStreaming ? (
                    <button
                      onClick={startStream}
                      className="flex items-center gap-2 bg-green-500 hover:bg-green-600 text-white px-6 py-3 rounded-lg font-semibold transition"
                    >
                      <Camera className="w-5 h-5" />
                      Iniciar Análisis
                    </button>
                  ) : (
                    <button
                      onClick={stopStream}
                      className="flex items-center gap-2 bg-red-500 hover:bg-red-600 text-white px-6 py-3 rounded-lg font-semibold transition"
                    >
                      <Square className="w-5 h-5" />
                      Detener Análisis
                    </button>
                  )}
                </div>
              </div>
            </div>

            {/* Stats Panel */}
            <div className="space-y-4">
              <div className="bg-white rounded-lg shadow-lg p-6">
                <div className="flex items-center gap-3 mb-4">
                  <Users className="w-6 h-6 text-indigo-600" />
                  <h2 className="text-xl font-semibold">
                    Estadísticas en Vivo
                  </h2>
                </div>

                <div className="space-y-4">
                  <div className="bg-indigo-50 rounded-lg p-4">
                    <p className="text-sm text-gray-600">
                      Estudiantes Detectados
                    </p>
                    <p className="text-3xl font-bold text-indigo-600">
                      {currentStats.total}
                    </p>
                  </div>

                  {isStreaming && currentStats.students.length > 0 && (
                    <div className="space-y-2 max-h-96 overflow-y-auto">
                      <h3 className="font-semibold text-gray-700 text-sm">
                        Atención Actual:
                      </h3>
                      {currentStats.students.map((student) => (
                        <div
                          key={student.id}
                          className="bg-gray-50 rounded p-3"
                        >
                          <div className="flex justify-between items-center">
                            <span className="text-sm font-medium">
                              {student.id}
                            </span>
                            <span
                              className={`text-sm font-bold ${
                                student.avg_attention >= 70
                                  ? "text-green-600"
                                  : student.avg_attention >= 50
                                  ? "text-yellow-600"
                                  : "text-red-600"
                              }`}
                            >
                              {student.avg_attention.toFixed(1)}%
                            </span>
                          </div>
                          <div className="mt-2 bg-gray-200 rounded-full h-2">
                            <div
                              className={`h-2 rounded-full transition-all ${
                                student.avg_attention >= 70
                                  ? "bg-green-500"
                                  : student.avg_attention >= 50
                                  ? "bg-yellow-500"
                                  : "bg-red-500"
                              }`}
                              style={{ width: `${student.avg_attention}%` }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        ) : (
          // Results View
          <div className="space-y-6">
            <div className="bg-white rounded-lg shadow-lg p-6">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <Award className="w-8 h-8 text-yellow-500" />
                  <h2 className="text-2xl font-bold">Ranking de Atención</h2>
                </div>
                <button
                  onClick={resetAnalysis}
                  className="bg-indigo-500 hover:bg-indigo-600 text-white px-4 py-2 rounded-lg transition"
                >
                  Nuevo Análisis
                </button>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {finalResults.ranking.map((student, index) => (
                  <div
                    key={student.id}
                    className={`rounded-lg p-6 border-2 ${
                      index === 0
                        ? "bg-yellow-50 border-yellow-400"
                        : index === 1
                        ? "bg-gray-50 border-gray-400"
                        : index === 2
                        ? "bg-orange-50 border-orange-400"
                        : "bg-white border-gray-200"
                    }`}
                  >
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex items-center gap-2">
                        <span
                          className={`text-2xl font-bold ${
                            index === 0
                              ? "text-yellow-600"
                              : index === 1
                              ? "text-gray-600"
                              : index === 2
                              ? "text-orange-600"
                              : "text-gray-400"
                          }`}
                        >
                          #{index + 1}
                        </span>
                        {index < 3 && (
                          <Award className="w-5 h-5 text-yellow-500" />
                        )}
                      </div>
                      <TrendingUp
                        className={`w-5 h-5 ${
                          student.attention_percentage >= 70
                            ? "text-green-500"
                            : student.attention_percentage >= 50
                            ? "text-yellow-500"
                            : "text-red-500"
                        }`}
                      />
                    </div>

                    {student.face_image && (
                      <div className="mb-3">
                        <img
                          src={student.face_image}
                          alt={student.id}
                          className="w-24 h-24 rounded-full mx-auto object-cover border-4 border-white shadow-md"
                        />
                      </div>
                    )}

                    <h3 className="font-semibold text-center mb-2">
                      {student.id}
                    </h3>

                    <div className="space-y-2">
                      <div className="text-center">
                        <p className="text-3xl font-bold text-indigo-600">
                          {student.attention_percentage.toFixed(1)}%
                        </p>
                        <p className="text-xs text-gray-500">
                          Atención Promedio
                        </p>
                      </div>

                      <div className="bg-gray-200 rounded-full h-3">
                        <div
                          className={`h-3 rounded-full ${
                            student.attention_percentage >= 70
                              ? "bg-green-500"
                              : student.attention_percentage >= 50
                              ? "bg-yellow-500"
                              : "bg-red-500"
                          }`}
                          style={{ width: `${student.attention_percentage}%` }}
                        />
                      </div>

                      <p className="text-xs text-gray-500 text-center">
                        Duración: {student.duration_seconds.toFixed(1)}s
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
