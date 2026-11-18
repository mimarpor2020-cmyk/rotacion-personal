"""Aplicación web: predicción de rotación de personal (HR Attrition).

Flujo de inferencia (idéntico al de la sección 14 del notebook):
    formulario -> DataFrame (30 columnas crudas) -> ingeniería de características
    -> preprocessor.transform -> model.predict -> umbral calibrado

Recomendaciones (LLM + aprendizaje por refuerzo):
    predicción -> agente RL elige la estrategia de prompt -> GPT-OSS 120B (Groq) redacta
    -> el usuario califica con estrellas -> recompensa -> actualiza Q(s, a)

Ejecutar:  python app.py          (servidor local en http://127.0.0.1:5000)
Túnel:     python tunel_ngrok.py  (servidor + enlace público)
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

import llm
import rl
from campos import CAMPOS, EJEMPLOS, ETIQUETAS, GRUPOS
from features import ingenieria_caracteristicas

BASE = os.path.dirname(os.path.abspath(__file__))
DIR_MODELO = os.environ.get("DIR_MODELO", os.path.join(BASE, "modelo"))
ARCHIVOS = {
    "modelo": os.path.join(DIR_MODELO, "modelo_attrition.keras"),
    "preprocesador": os.path.join(DIR_MODELO, "preprocessor.pkl"),
    "config": os.path.join(DIR_MODELO, "config_modelo.json"),
}
# Pesos exportados con exportar_pesos.py: permiten predecir con NumPy, sin TensorFlow (hosting gratis)
PESOS_NUMPY = os.path.join(DIR_MODELO, "pesos_red.npz")

app = Flask(__name__)
AGENTE = rl.Agente(rl.RUTA_BD)


# ------------------------------------------------------------------
# Carga de artefactos (una sola vez, al iniciar el servidor)
# ------------------------------------------------------------------
def cargar_artefactos():
    requeridos = dict(ARCHIVOS)
    if os.path.exists(PESOS_NUMPY):
        requeridos.pop("modelo")           # con los pesos en NumPy no hace falta el .keras
    faltan = [os.path.basename(r) for r in requeridos.values() if not os.path.exists(r)]
    if faltan:
        return None, "Faltan archivos en app_web/modelo/: " + ", ".join(faltan) + \
            ". Cópialos desde el notebook (sección 16) y reinicia el servidor."
    with open(ARCHIVOS["config"], encoding="utf-8") as f:
        config = json.load(f)
    if os.path.exists(PESOS_NUMPY):
        from red_numpy import RedNumpy
        modelo = RedNumpy(PESOS_NUMPY)
    else:
        try:
            import keras                   # Keras 3 (TensorFlow >= 2.16)
        except ImportError:
            from tensorflow import keras   # versiones anteriores
        modelo = keras.models.load_model(ARCHIVOS["modelo"], compile=False)
    prep = joblib.load(ARCHIVOS["preprocesador"])

    aviso = None
    try:
        import sklearn
        v_entreno = config.get("versiones", {}).get("scikit-learn")
        if v_entreno and v_entreno != sklearn.__version__:
            aviso = (f"El preprocesador se guardó con scikit-learn {v_entreno} y aquí está instalado "
                     f"{sklearn.__version__}. Si ves errores, instala la misma versión.")
    except ImportError:
        pass
    return {"modelo": modelo, "prep": prep, "config": config, "aviso": aviso}, None


ART, ERROR_CARGA = cargar_artefactos()
if ERROR_CARGA:
    print("AVISO:", ERROR_CARGA)
else:
    print(f"Modelo cargado: {ART['config']['nombre_modelo']} | umbral {ART['config']['umbral_decision']:.2f}")
print("LLM:", f"{llm.nombre_modelo()} vía Groq" if llm.disponible() else "sin GROQ_API_KEY (recomendaciones desactivadas)")


# ------------------------------------------------------------------
# Lógica de predicción
# ------------------------------------------------------------------
def valores_por_defecto():
    if ART:
        return dict(ART["config"]["valores_por_defecto"])
    # Valores típicos del dataset (solo para mostrar el formulario si falta el modelo)
    return {"Age": 36, "Gender": "Male", "MaritalStatus": "Married", "DistanceFromHome": 7, "Education": 3,
            "EducationField": "Life Sciences", "Department": "Research & Development", "JobRole": "Sales Executive",
            "JobLevel": 2, "OverTime": "No", "BusinessTravel": "Travel_Rarely", "JobInvolvement": 3,
            "PerformanceRating": 3, "MonthlyIncome": 4900, "PercentSalaryHike": 14, "StockOptionLevel": 1,
            "DailyRate": 800, "HourlyRate": 66, "MonthlyRate": 14200, "JobSatisfaction": 3,
            "EnvironmentSatisfaction": 3, "RelationshipSatisfaction": 3, "WorkLifeBalance": 3,
            "TotalWorkingYears": 10, "NumCompaniesWorked": 2, "YearsAtCompany": 5, "YearsInCurrentRole": 3,
            "YearsSinceLastPromotion": 1, "YearsWithCurrManager": 3, "TrainingTimesLastYear": 3}


def rango(nombre):
    if ART:
        return ART["config"]["rangos_numericos"].get(nombre)
    return None


def validar(datos):
    """Convierte y valida los datos del formulario. Devuelve (registro, errores, advertencias)."""
    registro, errores, advertencias = {}, {}, []
    for c in CAMPOS:
        n = c["nombre"]
        crudo = datos.get(n)
        if crudo is None or str(crudo).strip() == "":
            errores[n] = "Completa este campo."
            continue
        if c["tipo"] == "sel":
            validas = {str(v): v for v, _ in c["opciones"]}
            if str(crudo) not in validas:
                errores[n] = "Elige una opción de la lista."
                continue
            registro[n] = validas[str(crudo)]
        else:
            try:
                valor = int(float(str(crudo).replace(",", ".")))
            except ValueError:
                errores[n] = "Escribe un número entero."
                continue
            if valor < 0:
                errores[n] = "No puede ser negativo."
                continue
            r = rango(n)
            if r and not (r[0] <= valor <= r[1]):
                advertencias.append(f"{c['etiqueta']} = {valor} está fuera del rango visto en el "
                                    f"entrenamiento ({r[0]}–{r[1]}); la predicción es menos fiable.")
            registro[n] = valor

    # Reglas de coherencia entre campos
    def g(k):
        return registro.get(k)
    if g("Age") is not None and g("Age") < 18:
        errores["Age"] = "La edad mínima es 18 años."
    if None not in (g("YearsAtCompany"), g("TotalWorkingYears")) and g("YearsAtCompany") > g("TotalWorkingYears"):
        errores["YearsAtCompany"] = "No puede superar los años de experiencia total."
    for k in ("YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"):
        if None not in (g(k), g("YearsAtCompany")) and g(k) > g("YearsAtCompany"):
            errores[k] = "No puede superar los años en esta empresa."
    if None not in (g("TotalWorkingYears"), g("Age")) and g("TotalWorkingYears") > g("Age") - 14:
        errores["TotalWorkingYears"] = "Demasiados años de experiencia para la edad indicada."
    return registro, errores, advertencias


def probabilidades(df_raw):
    Xp = ART["prep"].transform(ingenieria_caracteristicas(df_raw))
    return ART["modelo"].predict(Xp, verbose=0).ravel()


OPCIONES = {c["nombre"]: dict((str(k), t) for k, t in c.get("opciones", [])) for c in CAMPOS}


def texto_valor(col, valor):
    return OPCIONES[col].get(str(valor), str(valor))


def describir_perfil(registro):
    """Perfil en lenguaje natural para el prompt: [(etiqueta, valor legible)]."""
    perfil = []
    for c in CAMPOS:
        v = texto_valor(c["nombre"], registro[c["nombre"]])
        perfil.append((c["etiqueta"], f"{v} {c['sufijo']}".strip() if c.get("sufijo") else v))
    return perfil


def predecir(registro):
    cfg = ART["config"]
    columnas = cfg["columnas_entrada"]
    umbral = cfg["umbral_decision"]
    margen = cfg["margen_confianza"]

    base = pd.DataFrame([registro])[columnas]
    # Fila 0 = empleado real. Filas 1..n = el mismo empleado con UNA variable puesta en su valor típico.
    # La diferencia de probabilidad muestra cuánto pesa esa variable en este caso concreto.
    tipico = cfg["valores_por_defecto"]
    variantes = [base]
    cambiadas = []
    for col in columnas:
        if registro[col] != tipico[col]:
            v = base.copy()
            v[col] = tipico[col]
            variantes.append(v)
            cambiadas.append(col)
    p = probabilidades(pd.concat(variantes, ignore_index=True))
    prob = float(p[0])

    factores = []
    for col, p_alt in zip(cambiadas, p[1:]):
        delta = prob - float(p_alt)
        if abs(delta) >= 0.005:
            factores.append({"variable": ETIQUETAS[col], "valor": texto_valor(col, registro[col]),
                             "delta": round(delta * 100, 1)})
    factores.sort(key=lambda f: -abs(f["delta"]))

    return {
        "probabilidad": round(prob, 4),
        "porcentaje": round(prob * 100, 1),
        "abandona": prob >= umbral,
        "prediccion": "Yes" if prob >= umbral else "No",
        "confianza": "alta" if abs(prob - umbral) >= margen else "baja",
        "umbral": umbral,
        "margen": margen,
        "factores": factores[:6],
    }


# ------------------------------------------------------------------
# Rutas
# ------------------------------------------------------------------
def contexto_base(valores, **extra):
    cfg = ART["config"] if ART else None
    return dict(campos=CAMPOS, grupos=GRUPOS, valores=valores, ejemplos=EJEMPLOS,
                rango=rango, error_carga=ERROR_CARGA, aviso=ART["aviso"] if ART else None,
                metricas=cfg["metricas_prueba"] if cfg else None,
                metricas_nuevos=cfg["metricas_datos_nuevos"] if cfg else None,
                umbral=cfg["umbral_decision"] if cfg else 0.5,
                margen=cfg["margen_confianza"] if cfg else 0.2,
                llm_activo=llm.disponible(), llm_nombre=llm.nombre_modelo(), **extra)


@app.route("/", methods=["GET", "POST"])
def inicio():
    valores = valores_por_defecto()
    resultado, errores, advertencias, ejemplo = None, {}, [], None

    if request.method == "GET" and request.args.get("ejemplo") in EJEMPLOS:
        ejemplo = request.args["ejemplo"]
        valores.update(EJEMPLOS[ejemplo][1])
        request_datos = valores
    elif request.method == "POST":
        request_datos = request.form.to_dict()
        valores.update(request_datos)
    else:
        request_datos = valores          # primera visita: se evalúa el perfil típico

    if request_datos is not None and ART:
        registro, errores, advertencias = validar(request_datos)
        if not errores:
            resultado = predecir(registro)

    return render_template("index.html", **contexto_base(valores, resultado=resultado, errores=errores,
                                                         advertencias=advertencias, ejemplo=ejemplo))


@app.route("/api/predecir", methods=["POST"])
def api_predecir():
    if not ART:
        return jsonify({"error": ERROR_CARGA}), 503
    datos = request.get_json(silent=True) or request.form.to_dict()
    registro, errores, advertencias = validar(datos)
    if errores:
        return jsonify({"errores": errores, "advertencias": advertencias}), 400
    r = predecir(registro)
    r["advertencias"] = advertencias
    return jsonify(r)


@app.route("/api/recomendar", methods=["POST"])
def api_recomendar():
    """Predicción -> el agente elige la estrategia -> el LLM redacta las recomendaciones."""
    if not ART:
        return jsonify({"error": ERROR_CARGA}), 503
    datos = request.get_json(silent=True) or request.form.to_dict()
    registro, errores, _ = validar(datos)
    if errores:
        return jsonify({"errores": errores}), 400
    r = predecir(registro)

    contexto = rl.contexto_de(r["probabilidad"], r["umbral"], r["margen"])
    estrategia, explorando, q = AGENTE.elegir(contexto)
    mensajes = llm.construir_mensajes(describir_perfil(registro), r, contexto, rl.ESTRATEGIAS[estrategia],
                                      AGENTE.retroalimentacion(contexto, estrategia))
    try:
        rec, modelo, segundos = llm.generar(mensajes)
    except llm.ErrorLLM as e:
        return jsonify({"error": str(e)}), 502

    rec_id = AGENTE.registrar(contexto, estrategia, explorando, r["probabilidad"], modelo, registro, rec)
    return jsonify({
        "id": rec_id,
        "recomendacion": rec,
        "caracteres": llm.caracteres(rec),
        "modelo": llm.nombre_modelo(modelo),
        "segundos": segundos,
        "contexto": contexto,
        "contexto_nombre": rl.CONTEXTOS[contexto],
        "estrategia": estrategia,
        "estrategia_nombre": rl.ESTRATEGIAS[estrategia]["nombre"],
        "explorando": explorando,
        "q": round(q, 3),
        "n": AGENTE.tabla()[1][(contexto, estrategia)],
        "porcentaje": r["porcentaje"],
    })


@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    """Calificación del usuario = recompensa del agente."""
    datos = request.get_json(silent=True) or {}
    try:
        rec_id, estrellas = int(datos.get("id")), int(datos.get("estrellas"))
    except (TypeError, ValueError):
        return jsonify({"error": "Faltan el id de la recomendación o las estrellas."}), 400
    if not 1 <= estrellas <= 5:
        return jsonify({"error": "La calificación va de 1 a 5 estrellas."}), 400
    comentario = str(datos.get("comentario") or "").strip()[:500]
    cambio = AGENTE.calificar(rec_id, estrellas, comentario)
    if cambio is None:
        return jsonify({"error": "No existe esa recomendación."}), 404
    cambio["estrategia_nombre"] = rl.ESTRATEGIAS[cambio["estrategia"]]["nombre"]
    cambio["preferida_nombre"] = rl.ESTRATEGIAS[cambio["preferida"]]["nombre"]
    cambio["contexto_nombre"] = rl.CONTEXTOS[cambio["contexto"]]
    return jsonify(cambio)


@app.route("/aprendizaje")
def aprendizaje():
    return render_template("aprendizaje.html", estado=AGENTE.estado(), estrategias=rl.ESTRATEGIAS,
                           llm_nombre=llm.nombre_modelo(), llm_activo=llm.disponible())


@app.route("/api/rl/estado")
def api_rl_estado():
    return jsonify(AGENTE.estado())


@app.route("/salud")
def salud():
    return jsonify({"estado": "ok" if ART else "sin_modelo", "detalle": ERROR_CARGA,
                    "llm": llm.nombre_modelo() if llm.disponible() else None})


if __name__ == "__main__":
    puerto = int(os.environ.get("PUERTO", 5000))
    app.run(host="0.0.0.0", port=puerto, debug=False)
