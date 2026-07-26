/* Monitor en vivo de un trabajo.
   Se conecta por WebSocket, pinta el snapshot inicial y luego aplica deltas.
   Si el WebSocket cae, recurre a sondear /eventos hasta poder reconectar. */

class MonitorDeTrabajo {
  constructor(raiz) {
    this.idTrabajo = raiz.dataset.idTrabajo;
    this.ultimaSecuencia = 0;
    this.socket = null;
    this.reintento = 1000;
    this.sondeo = null;
    this.terminado = false;
    this.claveActiva = null;
    this.corridas = [];

    this.el = {
      descripcion: document.getElementById("descripcion-trabajo"),
      meta: document.getElementById("meta-trabajo"),
      insignia: document.getElementById("insignia-estado"),
      cancelar: document.getElementById("btn-cancelar"),
      resultados: document.getElementById("enlace-resultados"),
      barraGlobal: document.getElementById("barra-global"),
      textoGlobal: document.getElementById("texto-global"),
      barraCorrida: document.getElementById("barra-corrida"),
      textoCorrida: document.getElementById("texto-corrida"),
      etiquetaCorrida: document.getElementById("etiqueta-corrida"),
      cuentaCorridas: document.getElementById("cuenta-corridas"),
      tabla: document.querySelector("#tabla-corridas tbody"),
      bitacora: document.getElementById("bitacora"),
    };

    this.grafica = new GraficaDeCurvas(document.getElementById("grafica-loss"), {
      etiquetaX: "época",
      etiquetaY: "pérdida",
    });
    if (!this.grafica.disponible) {
      document.getElementById("sin-grafica").style.display = "block";
    }

    this.el.cancelar.addEventListener("click", () => this.cancelar());

    const selectorEscala = document.getElementById("escala-y");
    if (selectorEscala) {
      selectorEscala.addEventListener("change", (e) =>
        this.grafica.cambiarEscala(e.target.value)
      );
    }

    this.conectar();
  }

  /* --- conexión --------------------------------------------------------- */

  conectar() {
    const url = ClienteAPI.urlWebSocket("/ws/trabajos/" + this.idTrabajo);
    this.socket = new WebSocket(url);

    this.socket.onopen = () => {
      this.reintento = 1000;
      this.detenerSondeo();
      if (this.ultimaSecuencia > 0) {
        this.socket.send(JSON.stringify({ accion: "replay", desde: this.ultimaSecuencia }));
      }
    };

    this.socket.onmessage = (mensaje) => {
      const evento = JSON.parse(mensaje.data);
      if (evento.tipo === "snapshot") this.aplicarSnapshot(evento);
      else this.aplicarEvento(evento);
    };

    this.socket.onclose = () => {
      if (this.terminado) return;
      /* Mientras no vuelva el WebSocket, se sondea la API. */
      this.iniciarSondeo();
      setTimeout(() => this.conectar(), this.reintento);
      this.reintento = Math.min(this.reintento * 2, 30000);
    };

    this.socket.onerror = () => this.socket.close();
  }

  iniciarSondeo() {
    if (this.sondeo) return;
    this.sondeo = setInterval(() => this.sondear(), 3000);
  }

  detenerSondeo() {
    if (this.sondeo) { clearInterval(this.sondeo); this.sondeo = null; }
  }

  async sondear() {
    try {
      const datos = await ClienteAPI.obtener(
        "/api/trabajos/" + this.idTrabajo + "/eventos?desde=" + this.ultimaSecuencia
      );
      datos.eventos.forEach((e) => this.aplicarEvento(e));
      this.grafica.refrescar();
    } catch (error) {
      /* El servidor no responde; se reintenta en el siguiente ciclo. */
    }
  }

  /* --- pintado ---------------------------------------------------------- */

  aplicarSnapshot(snapshot) {
    const trabajo = snapshot.trabajo;
    this.ultimaSecuencia = snapshot.secuencia || 0;

    this.el.descripcion.textContent = trabajo.descripcion;
    this.el.descripcion.classList.remove("tenue");
    this.el.meta.textContent =
      trabajo.id + " · " + (trabajo.metadatos.dispositivo || "cpu") +
      (trabajo.metadatos.minilotes ? " · minilotes" : "");

    this.actualizarEstado(trabajo);
    this.fijarGlobal(trabajo.fraccion);

    this.grafica.limpiar();
    Object.keys(snapshot.curvas || {}).forEach((clave) => {
      const curva = snapshot.curvas[clave];
      this.grafica.fijarSerie(clave, curva.epocas, curva.loss, this.etiquetaDe(clave), false);
    });
    this.grafica.refrescar();

    this.corridas = snapshot.corridas_terminadas || [];
    this.pintarCorridas();

    if (snapshot.bitacora && snapshot.bitacora.length) {
      this.el.bitacora.textContent = snapshot.bitacora.join("\n");
      this.el.bitacora.scrollTop = this.el.bitacora.scrollHeight;
    }
  }

  aplicarEvento(evento) {
    if (evento.secuencia) {
      /* Los eventos que ya se aplicaron (tras un replay) se ignoran. */
      if (evento.secuencia <= this.ultimaSecuencia) return;
      this.ultimaSecuencia = evento.secuencia;
    }

    switch (evento.tipo) {
      case "epoca": {
        const clave = evento.optimizador + "|" + evento.regla + "|" + evento.corrida;
        if (clave !== this.claveActiva) {
          this.claveActiva = clave;
          this.grafica.destacar(clave);
        }
        this.grafica.agregarPunto(clave, evento.epoca, evento.loss, this.etiquetaDe(clave));
        this.fijarCorrida(evento);
        this.fijarGlobal(evento.fraccion_global);
        this.programarRefresco();
        break;
      }
      case "corrida_inicio":
        this.el.etiquetaCorrida.textContent =
          evento.optimizador + " · " + evento.regla + " reglas · corrida " + (evento.corrida + 1);
        break;
      case "corrida_fin":
        this.corridas.push(Object.assign(
          { optimizador: evento.optimizador, regla: evento.regla, corrida: evento.corrida },
          evento.datos || {}
        ));
        this.pintarCorridas();
        this.fijarGlobal(evento.fraccion_global);
        break;
      case "bitacora":
        this.agregarBitacora(evento.datos.linea);
        break;
      case "estado":
        this.actualizarEstado({ estado: evento.datos.estado, error: evento.datos.error });
        break;
      case "error":
        this.agregarBitacora("⚠ " + (evento.datos ? evento.datos.mensaje : "error"));
        break;
      case "fin":
        this.fijarGlobal(evento.fraccion_global);
        this.terminado = true;
        if (evento.datos && evento.datos.ruta_resultado) {
          this.el.resultados.href = "/historial/web/" + this.idTrabajo;
          this.el.resultados.style.display = "inline-block";
        }
        break;
      case "latido":
      case "pong":
        break;
    }
  }

  /* Chart.js se refresca como mucho cada 200 ms, no en cada época. */
  programarRefresco() {
    if (this.refrescoPendiente) return;
    this.refrescoPendiente = setTimeout(() => {
      this.refrescoPendiente = null;
      this.grafica.refrescar();
    }, 200);
  }

  etiquetaDe(clave) {
    const partes = clave.split("|");
    return partes[0] + " · r" + partes[1] + " · #" + (parseInt(partes[2], 10) + 1);
  }

  fijarGlobal(fraccion) {
    if (fraccion === undefined || fraccion === null) return;
    const porcentaje = Math.max(0, Math.min(100, fraccion * 100));
    this.el.barraGlobal.style.width = porcentaje + "%";
    this.el.textoGlobal.textContent = porcentaje.toFixed(1) + " %";
  }

  fijarCorrida(evento) {
    const total = evento.epocas_totales || 1;
    const porcentaje = Math.max(0, Math.min(100, (evento.epoca / total) * 100));
    this.el.barraCorrida.style.width = porcentaje + "%";
    this.el.textoCorrida.textContent =
      "época " + evento.epoca + "/" + total + " · pérdida " + formatearNumero(evento.loss);
  }

  actualizarEstado(trabajo) {
    const estado = trabajo.estado || "pendiente";
    this.el.insignia.textContent = estado;
    this.el.insignia.className = "insignia " + estado;
    this.el.cancelar.disabled = !(estado === "pendiente" || estado === "ejecutando");
    if (estado === "cancelando") this.el.cancelar.textContent = "cancelando…";
    if (trabajo.error) this.agregarBitacora("⚠ " + trabajo.error);
    if (estado === "terminado" || estado === "cancelado" || estado === "fallido") {
      this.terminado = true;
    }
  }

  pintarCorridas() {
    this.el.cuentaCorridas.textContent = this.corridas.length ? "(" + this.corridas.length + ")" : "";
    if (!this.corridas.length) return;

    this.el.tabla.innerHTML = this.corridas
      .slice()
      .reverse()
      .map((c) => {
        const evaluacion = c.evaluacion || {};
        const texto = Object.keys(evaluacion).length
          ? Object.keys(evaluacion)
              .map((k) => k + " " + formatearNumero(evaluacion[k], 4))
              .join(" · ")
          : "—";
        return (
          "<tr>" +
          "<td>" + escaparHTML(c.optimizador) + "</td>" +
          "<td class='num'>" + c.regla + "</td>" +
          "<td class='num'>" + (c.corrida + 1) + "</td>" +
          "<td class='num'>" + (c.epocas || 0) + "</td>" +
          "<td class='num'>" + formatearNumero(c.loss_final) + "</td>" +
          "<td>" + texto + "</td>" +
          "<td class='num'>" + (c.duracion ? c.duracion.toFixed(1) + " s" : "—") + "</td>" +
          "<td class='tenue'>" + escaparHTML(c.motivo || "") + "</td>" +
          "</tr>"
        );
      })
      .join("");
  }

  agregarBitacora(linea) {
    if (!linea) return;
    const previo = this.el.bitacora.textContent;
    const base = previo === "(sin salida todavía)" ? "" : previo + "\n";
    const lineas = (base + linea).split("\n").slice(-300);
    this.el.bitacora.textContent = lineas.join("\n");
    this.el.bitacora.scrollTop = this.el.bitacora.scrollHeight;
  }

  /* --- acciones --------------------------------------------------------- */

  async cancelar() {
    this.el.cancelar.disabled = true;
    this.el.cancelar.textContent = "cancelando…";
    try {
      const trabajo = await ClienteAPI.enviar("/api/trabajos/" + this.idTrabajo + "/cancelar");
      this.actualizarEstado(trabajo);
    } catch (error) {
      this.agregarBitacora("⚠ No se pudo cancelar: " + error.message);
      this.el.cancelar.disabled = false;
      this.el.cancelar.textContent = "Cancelar";
    }
  }
}

document.addEventListener("DOMContentLoaded", function () {
  const raiz = document.querySelector(".monitor");
  if (raiz) new MonitorDeTrabajo(raiz);
});
