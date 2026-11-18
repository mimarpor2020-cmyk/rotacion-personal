#!/usr/bin/env bash
# Alternativa sin cuenta: localtunnel (requiere Node.js)
# 1) En otra terminal:  python app.py
# 2) Ejecuta este script: bash tunel_localtunnel.sh
# Al abrir el enlace, localtunnel pide una "Tunnel Password": es la IP pública del
# computador que ejecuta el servidor; se consulta en https://loca.lt/mytunnelpassword
PUERTO="${PUERTO:-5000}"
echo "Contraseña del túnel (IP pública): $(curl -s https://loca.lt/mytunnelpassword)"
npx --yes localtunnel --port "$PUERTO"
