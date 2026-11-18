// Predicción en tiempo real: cada cambio del formulario llama a /api/predecir
(function () {
  document.documentElement.classList.add("js");
  const form = document.getElementById("formulario");
  const caja = document.getElementById("resultado");
  if (!form || !caja) return;

  const $ = (id) => document.getElementById(id);
  let temporizador = null;
  let peticion = 0;

  function datosFormulario() {
    const d = {};
    new FormData(form).forEach((v, k) => { d[k] = v; });
    return d;
  }

  function limpiarErrores() {
    form.querySelectorAll(".campo").forEach((c) => {
      c.classList.remove("con-error");
      const ctrl = c.querySelector("input, select");
      if (ctrl) ctrl.removeAttribute("aria-invalid");
      const e = c.querySelector(".error");
      if (e) e.textContent = "";
    });
  }

  function mostrarErrores(errores) {
    Object.entries(errores).forEach(([campo, msj]) => {
      const c = form.querySelector(`[data-campo="${campo}"]`);
      if (!c) return;
      c.classList.add("con-error");
      const ctrl = c.querySelector("input, select");
      if (ctrl) ctrl.setAttribute("aria-invalid", "true");
      c.querySelector(".error").textContent = msj;
    });
  }

  function texto(el, t) { el.textContent = t; }

  function pintar(r) {
    caja.classList.remove("vacio");
    texto($("prob-num"), r.porcentaje.toLocaleString("es-CO", { minimumFractionDigits: 1, maximumFractionDigits: 1 }));
    caja.style.setProperty("--p", Math.min(Math.max(r.porcentaje, 0), 100));
    caja.querySelector(".instrumento").dataset.clase = r.abandona ? "si" : "no";
    texto($("veredicto"), r.abandona ? "El modelo predice que se irá."
                                     : "El modelo predice que permanecerá en la empresa.");
    texto($("confianza"), r.confianza === "alta"
      ? "Confianza alta: la probabilidad está lejos del umbral."
      : "Confianza baja: el caso está cerca del umbral; conviene revisarlo con Recursos Humanos.");

    const ul = $("factores");
    ul.innerHTML = "";
    if (!r.factores.length) {
      const li = document.createElement("li");
      li.className = "neutro";
      li.textContent = "Es un perfil típico: ningún dato se aparta del promedio.";
      ul.appendChild(li);
    }
    const tope = Math.max(...r.factores.map((f) => Math.abs(f.delta)), 0.1);
    r.factores.forEach((f) => {
      const li = document.createElement("li");
      li.className = f.delta > 0 ? "sube" : "baja";
      const a = document.createElement("span");
      a.className = "f-texto";
      a.append(f.variable + ": ");
      const b = document.createElement("strong");
      b.textContent = f.valor;
      a.appendChild(b);
      const d = document.createElement("span");
      d.className = "f-delta";
      d.textContent = (f.delta > 0 ? "+" : "") + f.delta.toLocaleString("es-CO", { minimumFractionDigits: 1, maximumFractionDigits: 1 }) + " pts";
      const barra = document.createElement("span");
      barra.className = "f-barra";
      barra.setAttribute("aria-hidden", "true");
      const relleno = document.createElement("span");
      relleno.style.width = (Math.abs(f.delta) / tope * 100) + "%";
      barra.appendChild(relleno);
      li.append(a, d, barra);
      ul.appendChild(li);
    });

    const adv = $("advertencias");
    adv.innerHTML = "";
    (r.advertencias || []).forEach((t) => {
      const li = document.createElement("li");
      li.textContent = t;
      adv.appendChild(li);
    });
  }

  async function predecir() {
    const id = ++peticion;
    caja.classList.add("cargando");
    try {
      const resp = await fetch("/api/predecir", {
        method: "POST",
        headers: { "Content-Type": "application/json", "ngrok-skip-browser-warning": "1" },
        body: JSON.stringify(datosFormulario()),
      });
      const r = await resp.json();
      if (id !== peticion) return;            // llegó una respuesta más nueva
      limpiarErrores();
      if (resp.ok) pintar(r);
      else if (r.errores) mostrarErrores(r.errores);
    } catch (e) {
      // Sin conexión con el servidor: el botón de envío normal sigue funcionando
      document.documentElement.classList.remove("js");
    } finally {
      if (id === peticion) caja.classList.remove("cargando");
    }
  }

  // ---------------- Recomendaciones del LLM + calificación (RL) ----------------
  const btnRec = $("btn-recomendar");
  const estadoIA = $("ia-estado");
  const rec = $("rec");
  const fb = $("feedback");
  const btnFb = $("btn-feedback");
  let recActual = null;
  const num = (x, d) => x.toLocaleString("es-CO", { minimumFractionDigits: d, maximumFractionDigits: d });
  const conSigno = (x) => (x > 0 ? "+" : "") + num(x, 2);

  function el(tag, clase, txt) {
    const e = document.createElement(tag);
    if (clase) e.className = clase;
    if (txt) e.textContent = txt;
    return e;
  }

  function pintarRecomendacion(r) {
    recActual = r;
    const d = r.recomendacion;
    const meta = $("rec-meta");
    meta.innerHTML = "";
    meta.append(el("span", "rec-estrategia", r.estrategia_nombre),
      el("span", "", r.explorando
        ? " El agente está explorando una estrategia distinta a la preferida."
        : r.n === 0
          ? ` El agente aún no ha probado esta estrategia en ${r.contexto_nombre.toLowerCase()}.`
          : ` Es la estrategia mejor valorada para ${r.contexto_nombre.toLowerCase()} (Q = ${num(r.q, 2)}, ${r.n} calificación${r.n === 1 ? "" : "es"}).`),
      el("span", "rec-modelo", ` Redactado por ${r.modelo} en ${num(r.segundos, 1)} s.`));
    texto($("rec-diag"), d.diagnostico);

    const ol = $("rec-acciones");
    ol.innerHTML = "";
    d.acciones.forEach((a) => {
      const li = el("li");
      const cab = el("p", "acc-cab");
      cab.appendChild(el("strong", "", a.titulo));
      if (a.plazo) cab.appendChild(el("span", "acc-plazo", a.plazo));
      li.appendChild(cab);
      if (a.detalle) li.appendChild(el("p", "acc-det", a.detalle));
      if (a.factor) li.appendChild(el("p", "acc-factor", "Factor: " + a.factor));
      ol.appendChild(li);
    });

    const ul = $("rec-preguntas");
    ul.innerHTML = "";
    d.preguntas.forEach((q) => ul.appendChild(el("li", "", q)));
    $("rec-preguntas-caja").hidden = !d.preguntas.length;
    texto($("rec-seg"), d.seguimiento);
    texto($("rec-cuenta"), `Texto generado: ${num(r.caracteres, 0)} caracteres.`);

    fb.reset();
    fb.querySelectorAll("input, textarea").forEach((c) => { c.disabled = false; });
    btnFb.disabled = true;
    texto($("fb-resultado"), "");
    rec.hidden = false;
    rec.classList.remove("desactualizada");
    // En escritorio el panel tiene su propio scroll: se desplaza solo el panel, no la página
    const panel = $("panel");
    const suave = matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth";
    if (panel.scrollHeight > panel.clientHeight) panel.scrollTo({ top: rec.offsetTop - 12, behavior: suave });
    else rec.scrollIntoView({ behavior: suave, block: "start" });
  }

  async function recomendar() {
    btnRec.disabled = true;
    texto(estadoIA, "Generando recomendaciones…");
    estadoIA.classList.remove("es-error");
    try {
      const resp = await fetch("/api/recomendar", {
        method: "POST",
        headers: { "Content-Type": "application/json", "ngrok-skip-browser-warning": "1" },
        body: JSON.stringify(datosFormulario()),
      });
      const r = await resp.json();
      if (resp.ok) {
        pintarRecomendacion(r);
        texto(estadoIA, "");
        btnRec.textContent = "Generar otra recomendación";
      } else {
        if (r.errores) { limpiarErrores(); mostrarErrores(r.errores); }
        texto(estadoIA, r.error || "Corrige los datos marcados en el formulario.");
        estadoIA.classList.add("es-error");
      }
    } catch (e) {
      texto(estadoIA, "No se pudo contactar al servidor.");
      estadoIA.classList.add("es-error");
    } finally {
      btnRec.disabled = false;
    }
  }

  async function calificar(ev) {
    ev.preventDefault();
    const elegida = fb.querySelector("input[name=estrellas]:checked");
    if (!recActual || !elegida) return;
    btnFb.disabled = true;
    const salida = $("fb-resultado");
    try {
      const resp = await fetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json", "ngrok-skip-browser-warning": "1" },
        body: JSON.stringify({ id: recActual.id, estrellas: Number(elegida.value), comentario: $("fb-comentario").value }),
      });
      const r = await resp.json();
      if (!resp.ok) { texto(salida, r.error); btnFb.disabled = false; return; }
      salida.innerHTML = "";
      const ecuacion = el("p", "fb-ecuacion");
      ecuacion.append(el("span", "fb-q", `Q(${r.contexto_nombre}, ${r.estrategia_nombre})`),
        el("strong", "fb-valores", `${num(r.q_antes, 2)} → ${num(r.q_despues, 2)}`));
      salida.append(
        el("p", "fb-gracias", `Calificación guardada. Recompensa r = ${conSigno(r.recompensa)}, α = ${num(r.alfa, 2)}.`),
        ecuacion,
        el("p", "", `Estrategia preferida ahora para ${r.contexto_nombre.toLowerCase()}: ${r.preferida_nombre}. `),
      );
      const a = el("a", "", "Ver lo que ha aprendido el agente");
      a.href = "/aprendizaje";
      salida.lastChild.appendChild(a);
      fb.querySelectorAll("input, textarea").forEach((c) => { c.disabled = true; });
    } catch (e) {
      texto(salida, "No se pudo enviar la calificación.");
      btnFb.disabled = false;
    }
  }

  if (btnRec) {
    btnRec.addEventListener("click", recomendar);
    fb.addEventListener("change", () => { btnFb.disabled = !fb.querySelector("input[name=estrellas]:checked"); });
    fb.addEventListener("submit", calificar);
  }

  form.addEventListener("input", () => {
    clearTimeout(temporizador);
    temporizador = setTimeout(predecir, 250);
    if (recActual && !rec.classList.contains("desactualizada")) {
      rec.classList.add("desactualizada");
      texto(estadoIA, "Cambiaste los datos: esta recomendación corresponde al perfil anterior.");
      estadoIA.classList.remove("es-error");
    }
  });
  form.addEventListener("submit", (ev) => { ev.preventDefault(); predecir(); });
})();
