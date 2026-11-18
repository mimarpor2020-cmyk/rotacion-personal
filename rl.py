"""Aprendizaje por refuerzo: agente que aprende qué estrategia de recomendación prefiere el usuario.

Formulación (bandido multibrazo contextual, Q-learning de un paso con γ = 0):
    Estado s     nivel de riesgo que da la red neuronal: bajo, duda, alto
    Acción a     estrategia de prompt con la que el LLM redacta la recomendación
    Recompensa r calificación del usuario: r = (estrellas - 3) / 2   ->  1★ = -1 ... 5★ = +1
    Política     ε-greedy: con probabilidad ε explora una estrategia al azar,
                 si no, explota la de mayor Q(s, a)
    Actualización  Q(s, a) <- Q(s, a) + α [r - Q(s, a)],   α = max(1 / N(s, a), α_min)

Los valores Q no se guardan: se recalculan reproduciendo las calificaciones en orden,
así la base de datos es la única fuente de verdad.

Reiniciar lo aprendido:  python rl.py --reiniciar
"""
import json
import os
import random
import sqlite3
import sys
import threading
from datetime import datetime

EPSILON = 0.2      # probabilidad de explorar
ALFA_MIN = 0.1     # tasa de aprendizaje mínima (permite seguir adaptándose)
Q_INICIAL = 0.5    # valor optimista: obliga a probar cada estrategia al menos una vez
RUTA_BD = os.environ.get("RUTA_BD", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "datos", "aprendizaje.db"))

CONTEXTOS = {
    "bajo": "Riesgo bajo",
    "duda": "Zona de duda",
    "alto": "Riesgo alto",
}

ESTRATEGIAS = {
    "acciones_directas": dict(
        nombre="Acciones directas",
        descripcion="Tres acciones breves, priorizadas por impacto.",
        instruccion=("Sé breve y concreto. Propón exactamente 3 acciones priorizadas por impacto; "
                     "cada 'detalle' en máximo dos frases. Deja 'preguntas' como lista vacía."),
    ),
    "plan_por_plazos": dict(
        nombre="Plan por plazos",
        descripcion="Plan temporal (inmediato, 30 y 90 días) con responsables.",
        instruccion=("Organiza 4 o 5 acciones como un plan en el tiempo. En 'plazo' usa solo "
                     "'Inmediato', '30 días' o '90 días', en ese orden. En cada 'detalle' indica quién "
                     "es responsable (jefe directo, Recursos Humanos o gerencia). Deja 'preguntas' vacía."),
    ),
    "conversacion": dict(
        nombre="Entrevista de permanencia",
        descripcion="Guion de conversación con el empleado y acciones según lo que responda.",
        instruccion=("Centra la recomendación en una entrevista de permanencia (stay interview) que el "
                     "jefe directo debe tener con el empleado. En 'preguntas' escribe 4 o 5 preguntas "
                     "abiertas y respetuosas, ligadas a los factores de riesgo. Luego propón 3 acciones "
                     "y explica en cada 'detalle' ante qué respuesta conviene aplicarla."),
    ),
    "ligado_a_factores": dict(
        nombre="Basada en los factores del modelo",
        descripcion="Cada acción ataca un factor concreto, empezando por el de mayor peso.",
        instruccion=("Propón 4 acciones; cada una debe atacar uno de los factores que más pesan, "
                     "empezando por el de mayor peso. En 'factor' cita el factor y su peso en puntos "
                     "(por ejemplo 'Hace horas extra: +12,0 pts'). En 'detalle' explica por qué la acción "
                     "reduciría ese factor. Deja 'preguntas' vacía."),
    ),
}


def contexto_de(probabilidad, umbral, margen):
    if probabilidad >= umbral + margen:
        return "alto"
    if probabilidad <= umbral - margen:
        return "bajo"
    return "duda"


def recompensa(estrellas):
    return (estrellas - 3) / 2


class Agente:
    def __init__(self, ruta_bd):
        self.ruta = ruta_bd
        self.candado = threading.Lock()
        os.makedirs(os.path.dirname(ruta_bd), exist_ok=True)
        with self._conexion() as cx:
            cx.execute("""CREATE TABLE IF NOT EXISTS recomendaciones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                creada TEXT NOT NULL,
                contexto TEXT NOT NULL,
                estrategia TEXT NOT NULL,
                explorando INTEGER NOT NULL,
                probabilidad REAL NOT NULL,
                modelo_llm TEXT,
                perfil TEXT,
                recomendacion TEXT,
                estrellas INTEGER,
                comentario TEXT,
                calificada TEXT)""")

    def _conexion(self):
        cx = sqlite3.connect(self.ruta)
        cx.row_factory = sqlite3.Row
        return cx

    def _calificaciones(self):
        with self._conexion() as cx:
            return cx.execute("""SELECT * FROM recomendaciones WHERE estrellas IS NOT NULL
                                 ORDER BY calificada, id""").fetchall()

    # ---------------- Valores Q ----------------
    def tabla(self):
        """Reproduce todas las calificaciones y devuelve (Q, N) por (contexto, estrategia)."""
        q = {(c, e): Q_INICIAL for c in CONTEXTOS for e in ESTRATEGIAS}
        n = {k: 0 for k in q}
        for f in self._calificaciones():
            k = (f["contexto"], f["estrategia"])
            if k not in q:
                continue                     # estrategia que ya no existe
            n[k] += 1
            alfa = max(1 / n[k], ALFA_MIN)
            q[k] += alfa * (recompensa(f["estrellas"]) - q[k])
        return q, n

    @staticmethod
    def _codiciosas(contexto, q, n):
        """Estrategias con el mayor Q; en empate gana la que tiene más calificaciones (más evidencia)."""
        clave = {e: (round(q[(contexto, e)], 9), n[(contexto, e)]) for e in ESTRATEGIAS}
        tope = max(clave.values())
        return [e for e in ESTRATEGIAS if clave[e] == tope]

    def mejor(self, contexto, q=None, n=None):
        if q is None or n is None:
            q, n = self.tabla()
        return self._codiciosas(contexto, q, n)[0]

    # ---------------- Política ε-greedy ----------------
    def elegir(self, contexto, rng=random):
        q, n = self.tabla()
        codiciosas = self._codiciosas(contexto, q, n)
        if rng.random() < EPSILON:
            elegida = rng.choice(list(ESTRATEGIAS))
        else:
            elegida = rng.choice(codiciosas)
        return elegida, elegida not in codiciosas, q[(contexto, elegida)]

    # ---------------- Memoria de retroalimentación ----------------
    def retroalimentacion(self, contexto, estrategia, limite=3):
        """Comentarios previos del usuario para esta estrategia y contexto (lo bien y lo mal valorado)."""
        with self._conexion() as cx:
            def consulta(condicion):
                return [f["comentario"] for f in cx.execute(
                    f"""SELECT comentario FROM recomendaciones
                        WHERE contexto = ? AND estrategia = ? AND {condicion}
                          AND comentario IS NOT NULL AND TRIM(comentario) != ''
                        ORDER BY calificada DESC LIMIT ?""", (contexto, estrategia, limite))]
            return {"bien": consulta("estrellas >= 4"), "mal": consulta("estrellas <= 2")}

    # ---------------- Registro ----------------
    def registrar(self, contexto, estrategia, explorando, probabilidad, modelo_llm, perfil, recomendacion):
        with self.candado, self._conexion() as cx:
            cur = cx.execute("""INSERT INTO recomendaciones
                (creada, contexto, estrategia, explorando, probabilidad, modelo_llm, perfil, recomendacion)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (ahora(), contexto, estrategia, int(explorando), probabilidad, modelo_llm,
                 json.dumps(perfil, ensure_ascii=False), json.dumps(recomendacion, ensure_ascii=False)))
            return cur.lastrowid

    def calificar(self, rec_id, estrellas, comentario):
        """Guarda la calificación y devuelve cómo cambió Q(s, a). None si la recomendación no existe."""
        with self.candado:
            with self._conexion() as cx:
                fila = cx.execute("SELECT contexto, estrategia FROM recomendaciones WHERE id = ?",
                                  (rec_id,)).fetchone()
            if not fila:
                return None
            k = (fila["contexto"], fila["estrategia"])
            q_antes = self.tabla()[0][k]
            with self._conexion() as cx:
                cx.execute("""UPDATE recomendaciones SET estrellas = ?, comentario = ?, calificada = ?
                              WHERE id = ?""", (estrellas, comentario or None, ahora(), rec_id))
            q, n = self.tabla()
        return {
            "contexto": k[0], "estrategia": k[1],
            "recompensa": recompensa(estrellas),
            "alfa": max(1 / n[k], ALFA_MIN),
            "q_antes": q_antes, "q_despues": q[k], "n": n[k],
            "preferida": self.mejor(k[0], q, n),
        }

    # ---------------- Resumen para la página de aprendizaje ----------------
    def estado(self):
        q, n = self.tabla()
        filas = self._calificaciones()
        curva, suma = [], 0.0
        for i, f in enumerate(filas, 1):
            suma += recompensa(f["estrellas"])
            curva.append(round(suma / i, 4))
        promedio_estrellas = {}
        for f in filas:
            promedio_estrellas.setdefault((f["contexto"], f["estrategia"]), []).append(f["estrellas"])
        with self._conexion() as cx:
            total = cx.execute("SELECT COUNT(*) FROM recomendaciones").fetchone()[0]
        recientes = sorted(filas, key=lambda f: f["calificada"], reverse=True)[:12]
        return {
            "parametros": {"epsilon": EPSILON, "alfa_min": ALFA_MIN, "q_inicial": Q_INICIAL},
            "contextos": [{
                "id": c, "nombre": nc, "preferida": self.mejor(c, q, n),
                "estrategias": [{
                    "id": e, "nombre": ESTRATEGIAS[e]["nombre"], "q": round(q[(c, e)], 4), "n": n[(c, e)],
                    "estrellas": (round(sum(promedio_estrellas[(c, e)]) / len(promedio_estrellas[(c, e)]), 2)
                                  if (c, e) in promedio_estrellas else None),
                } for e in ESTRATEGIAS],
            } for c, nc in CONTEXTOS.items()],
            "curva": curva,
            "generadas": total,
            "calificadas": len(filas),
            "recientes": [{
                "fecha": f["calificada"][:16].replace("T", " "),
                "contexto": CONTEXTOS.get(f["contexto"], f["contexto"]),
                "estrategia": ESTRATEGIAS.get(f["estrategia"], {}).get("nombre", f["estrategia"]),
                "explorando": bool(f["explorando"]), "estrellas": f["estrellas"],
                "comentario": f["comentario"] or "",
            } for f in recientes],
        }


def ahora():
    return datetime.now().isoformat(timespec="seconds")


if __name__ == "__main__":
    if "--reiniciar" in sys.argv:
        if os.path.exists(RUTA_BD):
            os.remove(RUTA_BD)
        print("Aprendizaje reiniciado: se borró", RUTA_BD)
    else:
        print(json.dumps(Agente(RUTA_BD).estado(), ensure_ascii=False, indent=2))
