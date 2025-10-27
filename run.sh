#!/bin/bash

# ============================================
# Script de inicio para el proyecto
# Sistema de Análisis de Atención en Clases
# ============================================

set -e

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Banner
echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════════════╗"
echo "║   Sistema de Análisis de Atención en Clases         ║"
echo "║   Inicializando el proyecto...                       ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Función para imprimir mensajes
print_message() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[i]${NC} $1"
}

# Verificar Python
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d " " -f 2)
        print_message "Python detectado: $PYTHON_VERSION"
        return 0
    else
        print_error "Python 3 no encontrado. Por favor instálalo."
        exit 1
    fi
}

# Verificar Node.js
check_node() {
    if command -v node &> /dev/null; then
        NODE_VERSION=$(node --version)
        print_message "Node.js detectado: $NODE_VERSION"
        return 0
    else
        print_error "Node.js no encontrado. Por favor instálalo."
        exit 1
    fi
}

# Verificar Docker (opcional)
check_docker() {
    if command -v docker &> /dev/null; then
        DOCKER_VERSION=$(docker --version | cut -d " " -f 3 | sed 's/,//')
        print_message "Docker detectado: $DOCKER_VERSION"
        return 0
    else
        print_warning "Docker no encontrado. Instalación local será usada."
        return 1
    fi
}

# Setup Backend
setup_backend() {
    print_info "Configurando Backend..."
    
    cd backend
    
    # Crear entorno virtual si no existe
    if [ ! -d "venv" ]; then
        print_info "Creando entorno virtual..."
        python3 -m venv venv
    fi
    
    # Activar entorno virtual
    print_info "Activando entorno virtual..."
    source venv/bin/activate
    
    # Instalar dependencias
    print_info "Instalando dependencias Python..."
    pip install --upgrade pip
    pip install -r requirements.txt
    
    # Crear archivo .env si no existe
    if [ ! -f ".env" ]; then
        print_info "Creando archivo .env desde .env.example..."
        cp .env.example .env
        print_warning "Por favor configura .env con tus valores"
    fi
    
    # Crear directorios necesarios
    mkdir -p data logs
    
    cd ..
    print_message "Backend configurado correctamente"
}

# Setup Frontend
setup_frontend() {
    print_info "Configurando Frontend..."
    
    cd frontend
    
    # Instalar dependencias
    if [ ! -d "node_modules" ]; then
        print_info "Instalando dependencias Node.js..."
        npm install
    else
        print_info "Actualizando dependencias Node.js..."
        npm update
    fi
    
    # Crear archivo .env si no existe
    if [ ! -f ".env" ]; then
        print_info "Creando archivo .env..."
        echo "REACT_APP_API_URL=http://localhost:8000" > .env
        echo "REACT_APP_WS_URL=ws://localhost:8000" >> .env
    fi
    
    cd ..
    print_message "Frontend configurado correctamente"
}

# Iniciar Backend
start_backend() {
    print_info "Iniciando Backend..."
    
    cd backend
    source venv/bin/activate
    
    # Iniciar en background
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 > ../logs/backend.log 2>&1 &
    BACKEND_PID=$!
    echo $BACKEND_PID > ../backend.pid
    
    cd ..
    print_message "Backend iniciado (PID: $BACKEND_PID)"
    print_info "Backend disponible en: http://localhost:8000"
}

# Iniciar Frontend
start_frontend() {
    print_info "Iniciando Frontend..."
    
    cd frontend
    
    # Iniciar en background
    npm start > ../logs/frontend.log 2>&1 &
    FRONTEND_PID=$!
    echo $FRONTEND_PID > ../frontend.pid
    
    cd ..
    print_message "Frontend iniciado (PID: $FRONTEND_PID)"
    print_info "Frontend disponible en: http://localhost:3000"
}

# Detener servicios
stop_services() {
    print_info "Deteniendo servicios..."
    
    if [ -f "backend.pid" ]; then
        BACKEND_PID=$(cat backend.pid)
        kill $BACKEND_PID 2>/dev/null || true
        rm backend.pid
        print_message "Backend detenido"
    fi
    
    if [ -f "frontend.pid" ]; then
        FRONTEND_PID=$(cat frontend.pid)
        kill $FRONTEND_PID 2>/dev/null || true
        rm frontend.pid
        print_message "Frontend detenido"
    fi
}

# Menú principal
show_menu() {
    echo ""
    echo -e "${BLUE}Selecciona una opción:${NC}"
    echo "1) Configurar proyecto (primera vez)"
    echo "2) Iniciar servicios"
    echo "3) Detener servicios"
    echo "4) Iniciar con Docker"
    echo "5) Limpiar instalación"
    echo "6) Ver logs"
    echo "7) Salir"
    echo ""
    read -p "Opción: " option
    
    case $option in
        1)
            check_python
            check_node
            setup_backend
            setup_frontend
            print_message "Proyecto configurado. Ahora puedes usar la opción 2 para iniciar."
            ;;
        2)
            start_backend
            sleep 3
            start_frontend
            print_message "Servicios iniciados correctamente"
            print_info "Presiona Ctrl+C para detener y volver al menú"
            wait
            ;;
        3)
            stop_services
            ;;
        4)
            if check_docker; then
                print_info "Iniciando con Docker..."
                docker-compose up --build
            fi
            ;;
        5)
            print_warning "Esto eliminará node_modules, venv y archivos generados"
            read -p "¿Estás seguro? (y/n): " confirm
            if [ "$confirm" = "y" ]; then
                rm -rf backend/venv backend/__pycache__ backend/data backend/logs
                rm -rf frontend/node_modules frontend/build
                rm -f backend.pid frontend.pid
                print_message "Limpieza completada"
            fi
            ;;
        6)
            mkdir -p logs
            echo -e "${BLUE}=== Backend Logs ===${NC}"
            tail -n 20 logs/backend.log 2>/dev/null || echo "No hay logs del backend"
            echo ""
            echo -e "${BLUE}=== Frontend Logs ===${NC}"
            tail -n 20 logs/frontend.log 2>/dev/null || echo "No hay logs del frontend"
            ;;
        7)
            stop_services
            print_message "¡Hasta luego!"
            exit 0
            ;;
        *)
            print_error "Opción inválida"
            ;;
    esac
}

# Manejar Ctrl+C
trap 'stop_services; exit' INT TERM

# Crear directorio de logs
mkdir -p logs

# Loop del menú
while true; do
    show_menu
done