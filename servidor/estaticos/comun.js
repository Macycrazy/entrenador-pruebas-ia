/* Utilidades compartidas por todas las vistas del panel. */

const Panel = {
  COLORES: ["#38bdf8", "#a78bfa", "#fbbf24", "#34d399", "#f87171", "#f472b6",
            "#60a5fa", "#c084fc", "#facc15", "#4ade80"],

  color(indice) { return this.COLORES[indice % this.COLORES.length]; },

  escapar(texto) {
    const d = document.createElement("div");
    d.textContent = texto ?? "";
    return d.innerHTML;
  },

  async estado(slug) {
    const r = await fetch(`/api/${slug}/estado`);
    const d = await r.json();
    return { ok: r.ok && d.listo, datos: d };
  },

  /** Pinta el chip de estado del modelo y devuelve si está listo. */
  async pintarEstado(slug, elemento, extra) {
    const { ok, datos } = await this.estado(slug);
    elemento.textContent = ok ? "modelo listo" : "sin modelo";
    elemento.className = "chip " + (ok ? "ok" : "mal");
    if (ok && extra) {
      const acc = datos.acc_val != null ? ` · val ${(datos.acc_val * 100).toFixed(1)}%` : "";
      extra.textContent = `${datos.arquitectura ?? datos.tarea} · ${datos.dispositivo}${acc}`;
      extra.hidden = false;
    }
    return { ok, datos };
  },

  async enviarArchivo(url, archivo, campo = "archivo", extra = {}) {
    const cuerpo = new FormData();
    cuerpo.append(campo, archivo, archivo.name || "captura");
    for (const [k, v] of Object.entries(extra)) cuerpo.append(k, v);
    const r = await fetch(url, { method: "POST", body: cuerpo });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `HTTP ${r.status}`);
    return r.json();
  },

  async enviarJson(url, datos) {
    const r = await fetch(url, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(datos),
    });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `HTTP ${r.status}`);
    return r.json();
  },

  /** Barras de probabilidad ordenadas de mayor a menor. */
  barras(probabilidades, colores = null) {
    return Object.entries(probabilidades)
      .sort((a, b) => b[1] - a[1])
      .map(([clase, p], i) => `
        <div class="barra"><span style="width:${(p * 100).toFixed(1)}%;
             background:${colores?.[clase] || this.color(i)}"></span></div>
        <div class="fila"><span>${this.escapar(clase)}</span>
             <span>${(p * 100).toFixed(1)}%</span></div>`)
      .join("");
  },
};

/** Cámara reutilizable: arranca, para, captura fotogramas y lista dispositivos. */
class Camara {
  constructor(video, { onFotograma = null, intervalo = 250 } = {}) {
    this.video = video;
    this.onFotograma = onFotograma;
    this.intervalo = intervalo;
    this.stream = null;
    this.temporizador = null;
    this.ocupado = false;
    this.lienzo = document.createElement("canvas");
    this.ctx = this.lienzo.getContext("2d");
  }

  get activa() { return this.stream !== null; }

  async iniciar(deviceId) {
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: { deviceId: deviceId ? { exact: deviceId } : undefined,
               width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    this.video.srcObject = this.stream;
    await this.video.play();
    if (this.onFotograma) this.arrancarBucle();
  }

  detener() {
    clearInterval(this.temporizador);
    this.temporizador = null;
    this.stream?.getTracks().forEach(t => t.stop());
    this.stream = null;
    this.video.srcObject = null;
  }

  arrancarBucle() {
    clearInterval(this.temporizador);
    this.temporizador = setInterval(() => this.tick(), this.intervalo);
  }

  async tick() {
    if (this.ocupado || !this.video.videoWidth) return;
    this.ocupado = true;
    try {
      const blob = await this.capturar();
      if (blob) await this.onFotograma(blob, this.lienzo.width, this.lienzo.height);
    } catch (e) {
      console.error(e);
    } finally {
      this.ocupado = false;
    }
  }

  capturar(ancho = 640, calidad = 0.82) {
    if (!this.video.videoWidth) return null;
    const alto = Math.round(this.video.videoHeight * (ancho / this.video.videoWidth));
    this.lienzo.width = ancho;
    this.lienzo.height = alto;
    this.ctx.drawImage(this.video, 0, 0, ancho, alto);
    return new Promise(res => this.lienzo.toBlob(res, "image/jpeg", calidad));
  }

  async listar(select) {
    const dispositivos = await navigator.mediaDevices.enumerateDevices();
    const camaras = dispositivos.filter(d => d.kind === "videoinput");
    select.hidden = camaras.length < 2;
    if (select.options.length === camaras.length) return;
    select.innerHTML = "";
    camaras.forEach((c, i) => {
      const o = document.createElement("option");
      o.value = c.deviceId;
      o.textContent = c.label || `Cámara ${i + 1}`;
      select.appendChild(o);
    });
  }
}

/** Dibuja cajas con etiqueta sobre un canvas superpuesto a un vídeo o imagen. */
function dibujarCajas(canvas, medio, cajas, anchoOrigen, altoOrigen,
                      { espejo = false, contener = true } = {}) {
  const ctx = canvas.getContext("2d");
  const rect = medio.getBoundingClientRect();
  canvas.width = rect.width;
  canvas.height = rect.height;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!anchoOrigen || !altoOrigen) return;

  const ajustar = contener ? Math.min : Math.max;
  const escala = ajustar(canvas.width / anchoOrigen, canvas.height / altoOrigen);
  const dx = (canvas.width - anchoOrigen * escala) / 2;
  const dy = (canvas.height - altoOrigen * escala) / 2;

  ctx.font = "600 15px system-ui, sans-serif";
  ctx.lineWidth = 3;

  for (const caja of cajas) {
    let px = caja.x * escala + dx;
    const py = caja.y * escala + dy;
    const pw = caja.ancho * escala, ph = caja.alto * escala;
    if (espejo) px = canvas.width - px - pw;

    ctx.strokeStyle = caja.color;
    ctx.strokeRect(px, py, pw, ph);

    const anchoTexto = ctx.measureText(caja.texto).width + 14;
    let yEtiqueta = py - 26;
    if (yEtiqueta < 0) yEtiqueta = py + ph + 2;
    if (yEtiqueta + 24 > canvas.height) yEtiqueta = py + 2;
    ctx.fillStyle = caja.color;
    ctx.fillRect(px, yEtiqueta, anchoTexto, 24);
    ctx.fillStyle = "#0b1020";
    ctx.fillText(caja.texto, px + 7, yEtiqueta + 17);
  }
}

/** Mensaje uniforme cuando la tarea no tiene modelo entrenado todavía. */
function avisoSinModelo(contenedor, vista) {
  // Callejón sin salida no: se ofrece el botón que abre la consola con esta tarea
  // ya seleccionada. El comando queda debajo, para quien prefiera la terminal.
  const cfg = Panel.escapar(vista.config);
  contenedor.innerHTML = `
    <div class="sin-modelo-aviso">
      <div class="titulo">Esta tarea todavía está sin entrenar</div>
      <p>El sistema sabe hacerla, pero aún no ha aprendido con datos. Entrenarla es un
         paso, y puedes verlo suceder en directo: el error bajando época tras época.</p>
      <div class="acciones-aviso">
        <a class="boton" href="/entrenamiento?config=${encodeURIComponent(vista.config)}">
          🚀 Entrenar esta tarea ahora</a>
        <a class="boton secundario"
           href="/entrenamiento?config=${encodeURIComponent(vista.config)}&amp;accion=estimar">
          ⏱️ Ver cuánto tardaría</a>
      </div>
      <details>
        <summary>Hacerlo desde la terminal</summary>
        <code>python entrenar.py --config ${cfg}</code>
        <code>python estimar_tiempo.py --config ${cfg}</code>
      </details>
    </div>`;
}

/* ---------------------------------------------------------------- gráficas */

/** Gráfica de líneas sobre canvas, sin librerías externas. */
function graficar(canvas, series, {ejeX = "época", maximoY = null, minimoY = null} = {}) {
  const ctx = canvas.getContext("2d");
  const escala = window.devicePixelRatio || 1;
  const ancho = canvas.clientWidth, alto = canvas.clientHeight;
  canvas.width = ancho * escala;
  canvas.height = alto * escala;
  ctx.setTransform(escala, 0, 0, escala, 0, 0);
  ctx.clearRect(0, 0, ancho, alto);

  const visibles = series.filter(s => s.valores.some(v => v != null && !isNaN(v)));
  if (!visibles.length) return;

  const margen = {arriba: 14, derecha: 12, abajo: 26, izquierda: 44};
  const anchoUtil = ancho - margen.izquierda - margen.derecha;
  const altoUtil = alto - margen.arriba - margen.abajo;
  const n = Math.max(...visibles.map(s => s.valores.length));
  const todos = visibles.flatMap(s => s.valores).filter(v => v != null && !isNaN(v));
  const alto_ = maximoY ?? Math.max(...todos);
  const bajo = minimoY ?? Math.min(...todos, 0);
  const rango = (alto_ - bajo) || 1;

  const x = i => margen.izquierda + (n > 1 ? (i / (n - 1)) * anchoUtil : anchoUtil / 2);
  const y = v => margen.arriba + altoUtil - ((v - bajo) / rango) * altoUtil;

  // rejilla y escala vertical
  ctx.strokeStyle = "#2a3355";
  ctx.fillStyle = "#98a3c4";
  ctx.font = "11px system-ui, sans-serif";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const valor = bajo + (rango * i) / 4;
    const py = y(valor);
    ctx.beginPath();
    ctx.moveTo(margen.izquierda, py);
    ctx.lineTo(ancho - margen.derecha, py);
    ctx.stroke();
    ctx.fillText(valor.toFixed(rango < 2 ? 2 : 0), 6, py + 4);
  }
  ctx.fillText(ejeX, ancho / 2 - 16, alto - 6);

  visibles.forEach((serie, indice) => {
    ctx.strokeStyle = serie.color || Panel.color(indice);
    ctx.lineWidth = 2;
    ctx.beginPath();
    let primero = true;
    serie.valores.forEach((valor, i) => {
      if (valor == null || isNaN(valor)) return;
      primero ? ctx.moveTo(x(i), y(valor)) : ctx.lineTo(x(i), y(valor));
      primero = false;
    });
    ctx.stroke();

    const ultimo = serie.valores.filter(v => v != null && !isNaN(v)).pop();
    if (ultimo != null) {
      ctx.fillStyle = ctx.strokeStyle;
      ctx.beginPath();
      ctx.arc(x(serie.valores.length - 1), y(ultimo), 3.5, 0, Math.PI * 2);
      ctx.fill();
    }
  });
}

/** Leyenda de una gráfica, como HTML. */
function leyenda(series) {
  return series.map((s, i) => `<span style="font-size:11.5px;color:var(--suave);
    margin-right:14px"><span style="display:inline-block;width:10px;height:3px;
    background:${s.color || Panel.color(i)};vertical-align:middle;margin-right:5px"></span>
    ${Panel.escapar(s.nombre)}</span>`).join("");
}

/* ---------------------------------------------------- ejemplos de un clic */
/* Quien viene a probar el sistema no trae fotos encima. Cada vista que pide un
   archivo recibe una tira de miniaturas: al pulsar una se rellena su propio
   <input type=file> y se dispara su "change", así que no hace falta tocar el
   código de cada vista. Si no hay ejemplos en esta máquina, no se muestra nada. */
async function montarEjemplos() {
  const slug = location.pathname.replace(/^\/|\/$/g, "");
  const entrada = document.querySelector("main input[type=file]#archivo");
  if (!slug || !entrada) return;

  // El botón "Probar con un ejemplo" no debe quedarse ahí sin hacer nada cuando
  // la máquina no tiene los datos descargados: se esconde.
  const rapido = document.getElementById("btn-ejemplo");
  let lista = [];
  try { lista = await (await fetch(`/api/${slug}/ejemplos`)).json(); } catch { lista = []; }
  if (!Array.isArray(lista) || !lista.length) {
    if (rapido) rapido.hidden = true;
    return;
  }

  const tira = document.createElement("div");
  tira.className = "tira-ejemplos";
  tira.innerHTML = `<span class="etq">Sin nada a mano, prueba con un ejemplo:</span>` +
    lista.map((e, i) =>
      `<button type="button" class="miniatura" data-i="${i}" title="${Panel.escapar(e.etiqueta)}">
         <img src="${e.url}" alt="${Panel.escapar(e.etiqueta)}" loading="lazy">
         <span>${Panel.escapar(e.etiqueta)}</span>
       </button>`).join("");

  // debajo de la barra de acciones si la hay; si no, justo antes del visor
  const ancla = document.querySelector("main .acciones-arriba") ||
                document.querySelector("main .controles");
  if (ancla) ancla.insertAdjacentElement("afterend", tira);
  else entrada.parentElement.insertAdjacentElement("beforebegin", tira);

  tira.querySelectorAll(".miniatura").forEach(boton => {
    boton.onclick = async () => {
      const i = Number(boton.dataset.i);
      tira.querySelectorAll(".miniatura").forEach(b => b.classList.remove("elegida"));
      boton.classList.add("elegida");
      const blob = await (await fetch(lista[i].url)).blob();
      const nombre = `ejemplo-${i}.${(blob.type.split("/")[1] || "jpg")}`;
      const dt = new DataTransfer();
      dt.items.add(new File([blob], nombre, {type: blob.type}));
      entrada.files = dt.files;
      entrada.dispatchEvent(new Event("change", {bubbles: true}));
    };
  });

  // El botón "Probar con un ejemplo" de la vista de cámara usa el primero.
  if (rapido) rapido.onclick = () => tira.querySelector(".miniatura")?.click();
}

document.addEventListener("DOMContentLoaded", montarEjemplos);
