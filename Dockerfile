# Imagen para publicar la app en Render (plan gratuito) u otro hosting con Docker.
# No instala TensorFlow: la red se evalúa con NumPy a partir de modelo/pesos_red.npz.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=10000

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY requirements-servidor.txt .
RUN pip install --no-cache-dir -r requirements-servidor.txt

COPY --chown=user:user . .
RUN mkdir -p datos && chown -R user:user /home/user/app

USER user
EXPOSE 10000

# Un solo proceso (el agente RL y la base SQLite se comparten) con varios hilos.
# Timeout amplio porque la llamada al LLM puede tardar unos segundos.
# Render define la variable PORT; si no existe se usa 10000.
CMD gunicorn --workers 1 --threads 4 --timeout 180 --bind 0.0.0.0:${PORT} app:app
