@echo off
REM Alternativa sin cuenta: localtunnel (requiere Node.js)
REM 1) En otra terminal:  python app.py
REM 2) Ejecuta este archivo. La "Tunnel Password" es tu IP publica: https://loca.lt/mytunnelpassword
if "%PUERTO%"=="" set PUERTO=5000
npx --yes localtunnel --port %PUERTO%
