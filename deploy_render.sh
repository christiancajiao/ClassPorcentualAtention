#!/bin/bash

# Script de deployment para Render.com
# Ubicación: raíz del proyecto

echo "🚀 Preparando deployment para Render..."

# Verificar que estamos en la raíz del proyecto
if [ ! -d "backend" ] || [ ! -d "frontend" ]; then
    echo "❌ Error: Ejecuta este script desde la raíz del proyecto"
    exit 1
fi

# 1. Verificar Git
if [ ! -d ".git" ]; then
    echo "📦 Inicializando repositorio Git..."
    git init
    git add .
    git commit -m "Initial commit for deployment"
else
    echo "✅ Repositorio Git encontrado"
fi

# 2. Crear .gitignore si no existe
if [ ! -f ".gitignore" ]; then
    echo "📝 Creando .gitignore..."
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*.so
.Python
venv/
ENV/
env/
*.egg-info/
.eggs/

# Node
node_modules/
build/
dist/
.env.local
.env.development.local
.env.test.local
.env.production.local

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Project specific
backend/data/
backend/logs/
*.log
.env
EOF
fi

# 3. Verificar requirements.txt
echo "📋 Verificando requirements.txt..."
cd backend
if [ ! -f "requirements.txt" ]; then
    echo "⚠️  Generando requirements.txt..."
    pip freeze > requirements.txt
fi
cd ..

# 4. Crear render.yaml si no existe
if [ ! -f "render.yaml" ]; then
    echo "📝 Creando render.yaml..."
    cat > render.yaml << 'EOF'
services:
  # Backend API
  - type: web
    name: atencion-backend
    env: python
    region: oregon
    plan: free
    branch: main
    buildCommand: |
      cd backend
      pip install --upgrade pip
      pip install -r requirements.txt
    startCommand: |
      cd backend
      uvicorn app.main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: 3.10.0
      - key: CORS_ORIGINS
        fromService:
          type: web
          name: atencion-frontend
          property: host
    healthCheckPath: /health
    
  # Frontend React
  - type: web
    name: atencion-frontend
    env: static
    region: oregon
    plan: free
    branch: main
    buildCommand: |
      cd frontend
      npm install
      npm run build
    staticPublishPath: ./frontend/build
    routes:
      - type: rewrite
        source: /*
        destination: /index.html
EOF
fi

# 5. Verificar estructura del proyecto
echo "🔍 Verificando estructura del proyecto..."

check_file() {
    if [ -f "$1" ]; then
        echo "  ✅ $1"
    else
        echo "  ❌ $1 (falta)"
        return 1
    fi
}

check_file "backend/requirements.txt"
check_file "backend/app/main.py"
check_file "frontend/package.json"
check_file "render.yaml"

# 6. Instrucciones finales
echo ""
echo "="*60
echo "✅ Proyecto preparado para deployment en Render"
echo "="*60
echo ""
echo "📋 Próximos pasos:"
echo ""
echo "1. Sube tu código a GitHub:"
echo "   git remote add origin https://github.com/TU_USUARIO/atencion-clases.git"
echo "   git branch -M main"
echo "   git add ."
echo "   git commit -m 'Preparado para deployment'"
echo "   git push -u origin main"
echo ""
echo "2. Ve a https://render.com y crea una cuenta"
echo ""
echo "3. En Render Dashboard:"
echo "   - Click 'New +' → 'Blueprint'"
echo "   - Conecta tu repositorio de GitHub"
echo "   - Render detectará render.yaml automáticamente"
echo "   - Click 'Apply'"
echo ""
echo "4. Espera ~15 minutos para el deployment"
echo ""
echo "5. Tus URLs serán:"
echo "   Backend:  https://atencion-backend.onrender.com"
echo "   Frontend: https://atencion-frontend.onrender.com"
echo ""
echo "="*60
echo ""
echo "📚 Para más detalles, lee DEPLOYMENT.md"
echo ""