# Aplicación web – Riesgo de rotación de personal

Aplicación Flask que recibe los datos de un empleado y muestra en tiempo real la
probabilidad de que deje la empresa, calculada por la red neuronal del notebook.
Sobre esa predicción, un LLM (GPT-OSS 120B de OpenAI) redacta recomendaciones para Recursos
Humanos, y un agente de aprendizaje por refuerzo aprende de las calificaciones del
usuario qué estilo de recomendación funciona mejor.

```
Red neuronal ──> probabilidad + factores ──> Agente RL elige estrategia ──> prompt ──> GPT-OSS 120B
      ^                                              ^                                    │
      │                                              └── recompensa  <── estrellas 1–5 <──┘
   formulario                                                         (usuario de RR. HH.)
```

```
app_web/
├── app.py                  Servidor Flask (formulario + API /api/predecir)
├── features.py             Ingeniería de características (idéntica al notebook)
├── llm.py                  Prompt + llamada a GPT-OSS 120B (Groq)
├── rl.py                   Agente de aprendizaje por refuerzo (bandido ε-greedy)
├── templates/aprendizaje.html  Página con la tabla Q y la curva de recompensa
├── datos/aprendizaje.db    Recomendaciones y calificaciones (se crea solo)
├── campos.py               Etiquetas del formulario y perfiles de ejemplo
├── templates/index.html    Interfaz (Jinja2)
├── static/style.css        Estilos
├── static/app.js           Actualización en tiempo real
├── tunel_ngrok.py          Servidor + enlace público con ngrok (pyngrok)
├── tunel_localtunnel.sh    Alternativa sin cuenta (localtunnel, Linux/macOS)
├── tunel_localtunnel.bat   Alternativa sin cuenta (localtunnel, Windows)
├── requirements.txt        Dependencias para correr en tu computador
├── requirements-servidor.txt  Dependencias fijas para el hosting (sin TensorFlow)
├── Dockerfile              Imagen para publicar en Render
├── red_numpy.py            La red neuronal evaluada con NumPy (mismo resultado que Keras)
├── exportar_pesos.py       Pasa los pesos del .keras a modelo/pesos_red.npz
└── modelo/                 <- aquí van los 3 archivos del notebook (+ pesos_red.npz)
```

## Publicar en internet gratis (Render)

El plan gratuito de Render tiene 512 MB de RAM, que no alcanzan para TensorFlow. Por eso
en el servidor la red se evalúa con NumPy: `exportar_pesos.py` copia los pesos entrenados
a `modelo/pesos_red.npz` y `red_numpy.py` aplica las mismas operaciones que Keras
(`relu(x·W + b)` en las capas ocultas y `sigmoid` en la salida; el Dropout no actúa al
predecir). Se verificó que la diferencia con Keras es menor a 1e-7. Si existe
`pesos_red.npz`, la app lo usa; si no, carga el `.keras` con TensorFlow como antes.
Si reentrenas el modelo, vuelve a ejecutar `python exportar_pesos.py`.

### Paso 1: subir el código a GitHub

1. Crea una cuenta en https://github.com y un repositorio nuevo, por ejemplo
   `rotacion-personal` (puede ser público o privado, sin README).
2. En esta carpeta ya hay un repositorio git con todo confirmado. Conéctalo y súbelo:

```bash
git remote add origin https://github.com/<tu-usuario>/rotacion-personal.git
git push -u origin main
```

   (La primera vez se abre una ventana para iniciar sesión en GitHub.)

### Paso 2: crear el servicio en Render

1. En https://dashboard.render.com → **New** → **Web Service**.
2. Conecta tu cuenta de GitHub y elige el repositorio `rotacion-personal`.
3. Configuración:
   - **Language**: Docker (Render lo detecta solo por el `Dockerfile`)
   - **Branch**: main · **Region**: la que quieras (Oregon u Ohio)
   - **Instance Type**: **Free**
   - **Environment Variables**: `GROQ_API_KEY` = tu clave `gsk_...`
4. **Deploy Web Service**. La construcción tarda 3–5 minutos. Cuando aparezca **Live**,
   el enlace es `https://rotacion-personal.onrender.com` (o parecido, arriba a la izquierda).

Notas:
- Con el plan gratis, si nadie entra en 15 minutos el servicio se duerme; la siguiente
  visita tarda unos 50 segundos en cargar. Ábrelo un minuto antes de la presentación.
- El disco del plan gratis no es permanente: cuando el servicio se reinicia o se vuelve
  a desplegar, las calificaciones vuelven al estado del `datos/aprendizaje.db` subido a
  GitHub. Para empezar en blanco: `python rl.py --reiniciar` antes de hacer el commit.
- Cada `git push` a `main` vuelve a desplegar la app automáticamente.

## 1. Copiar el modelo

Ejecuta el notebook completo en Colab. Al final se descarga `artefactos_modelo.zip`.
Copia estos tres archivos a `app_web/modelo/`:

- `modelo_attrition.keras`
- `preprocessor.pkl`
- `config_modelo.json`

## 2. Opción A: publicar desde Google Colab (la más sencilla)

1. Comprime la carpeta `app_web` como `app_web.zip` y súbela al panel de archivos de Colab.
2. Guarda tu authtoken de ngrok en los **Secretos de Colab** (icono de llave) con el nombre `NGROK_AUTHTOKEN`.
3. En la sección 17 del notebook cambia `EJECUTAR_DESPLIEGUE = True` y ejecuta la celda.
4. Copia el enlace `https://….ngrok-free.app` que aparece y compártelo con el profesor.

El enlace funciona mientras la sesión de Colab siga abierta.

## 3. Opción B: en tu computador

Requiere Python 3.10–3.12.

```bash
cd app_web
python -m venv venv
# Windows: venv\Scripts\activate      Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
pip install scikit-learn==<versión que aparece en modelo/config_modelo.json>
python app.py                 # http://127.0.0.1:5000
```

### Enlace público con ngrok

1. Crea una cuenta gratis en https://dashboard.ngrok.com/signup y copia tu authtoken.
2. Define la variable y lanza el túnel:

```bash
# Windows PowerShell
$env:NGROK_AUTHTOKEN="tu_token"
$env:GROQ_API_KEY="gsk_..."      # para las recomendaciones del LLM
# Linux / macOS
export NGROK_AUTHTOKEN=tu_token

python tunel_ngrok.py
```

El script arranca Flask y abre el túnel en el mismo proceso; en consola aparece el
enlace público. `Ctrl + C` detiene ambos.

Al abrir el enlace por primera vez, ngrok (plan gratuito) muestra una página de aviso:
hay que pulsar **Visit Site**.

### Alternativa sin cuenta: localtunnel

Requiere Node.js. Con `python app.py` corriendo en otra terminal:

```bash
bash tunel_localtunnel.sh       # Windows: tunel_localtunnel.bat
```

localtunnel pide una *Tunnel Password*: es la IP pública del computador que corre el
servidor (se ve en https://loca.lt/mytunnelpassword).

## LLM: recomendaciones generativas

**Modelo elegido: GPT-OSS 120B (OpenAI), servido por Groq.** Es el modelo de lenguaje
pre-entrenado de pesos abiertos de OpenAI, de la familia de ChatGPT: 120 mil millones de
parámetros con arquitectura de mezcla de expertos, ajustado para seguir instrucciones y
razonar antes de responder, con buen desempeño en español. Groq lo ejecuta en hardware
especializado (LPU), así que cada recomendación tarda unos dos segundos y el plan gratuito
alcanza para el proyecto. Se conecta por una API compatible con OpenAI y, si el modelo
principal falla, la app usa Qwen 3.8 27B (Alibaba) como respaldo.

Pipeline (`llm.py`):

1. La red neuronal entrega la probabilidad, el veredicto, la confianza y los factores que
   más pesan en ese empleado.
2. Esos datos, el perfil completo en español y la estrategia que eligió el agente RL forman
   el prompt. El mensaje de sistema fija el rol (consultor de gestión humana) y reglas
   éticas: nada de acciones basadas en género, edad o estado civil, y nada de despidos.
3. El LLM responde en modo JSON: diagnóstico, acciones (título, detalle, plazo, factor que
   ataca), preguntas para una conversación y un indicador de seguimiento.

Clave gratis: https://console.groq.com/keys → variable `GROQ_API_KEY`. Sin ella la
predicción sigue funcionando y la sección de recomendaciones avisa que falta la clave.
`GROQ_MODELO` permite cambiar de modelo.

## Aprendizaje por refuerzo: calificación de las recomendaciones

El usuario califica cada recomendación con 1 a 5 estrellas (y un comentario opcional).
`rl.py` lo modela como un **bandido multibrazo contextual**, un Q-learning de un solo paso
(γ = 0):

| Elemento | En la app |
|---|---|
| Estado *s* | Nivel de riesgo de la red: bajo (< 30 %), zona de duda, alto (> 70 %) |
| Acción *a* | Estrategia de prompt: acciones directas, plan por plazos, entrevista de permanencia, basada en factores |
| Recompensa *r* | `(estrellas − 3) / 2`, de −1 a +1 |
| Política | ε-greedy, ε = 0,2 (20 % explora, 80 % usa la mejor) |
| Actualización | `Q(s,a) ← Q(s,a) + α [r − Q(s,a)]`, α = máx(1/N, 0,1); Q₀ = 0,5 optimista |

Además, los comentarios de las recomendaciones bien y mal valoradas se incluyen en el
siguiente prompt de esa estrategia y nivel de riesgo, de modo que el texto también mejora
con la retroalimentación.

La página **/aprendizaje** muestra la tabla Q, la estrategia preferida en cada nivel de
riesgo, la curva de recompensa promedio acumulada y las últimas calificaciones con sus
comentarios.

Para borrar lo aprendido (por ejemplo, antes de la presentación): `python rl.py --reiniciar`.

## API

`POST /api/predecir` con un JSON que tenga las 30 variables crudas del dataset
(mismos nombres de columna). Respuesta:

```json
{
  "probabilidad": 0.565, "porcentaje": 56.5, "prediccion": "Yes", "abandona": true,
  "confianza": "baja", "umbral": 0.5, "margen": 0.22,
  "factores": [{"variable": "Hace horas extra", "valor": "Sí", "delta": 12.0}],
  "advertencias": []
}
```

Si hay datos inválidos o incoherentes (por ejemplo, más años en la empresa que de
experiencia total) responde `400` con `{"errores": {campo: mensaje}}`.

`POST /api/recomendar` recibe el mismo JSON y devuelve la recomendación del LLM, junto
con la estrategia elegida por el agente (`estrategia`, `explorando`, `q`) y su `id`.

`POST /api/feedback` con `{"id": 7, "estrellas": 4, "comentario": "..."}` registra la
recompensa y devuelve `recompensa`, `q_antes`, `q_despues`, `alfa` y la estrategia `preferida`.

`GET /api/rl/estado` devuelve la tabla Q y la curva de aprendizaje en JSON.

`GET /salud` indica si el modelo está cargado y qué LLM está activo.

## Cómo leer el resultado

- **Probabilidad**: salida sigmoide de la red.
- **Umbral**: punto de corte calibrado en validación (no necesariamente 50 %).
- **Zona de duda**: franja alrededor del umbral (± margen calibrado en la sección 15.4
  del notebook). Dentro de ella la confianza es baja y conviene revisión humana.
- **Qué pesa más**: cuánto cambia la probabilidad si ese dato tuviera el valor típico
  de la empresa (mediana o moda del entrenamiento). Es la sensibilidad del modelo para
  ese empleado, no una relación causal.
