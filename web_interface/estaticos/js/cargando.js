/* Indicador de carga con progreso.

   Un barrido completo produce cientos de curvas, y traerlas todas de golpe deja
   la página en blanco varios segundos sin explicar por qué. Este componente
   cubre la pantalla mientras tanto y dice qué se está haciendo y cuánto falta.

   Se usa así:

     const carga = new IndicadorDeCarga("Cargando el experimento");
     carga.fase("Descargando curvas…");
     carga.progreso(hechas, total, "curvas");
     carga.cerrar();
*/

class IndicadorDeCarga {
  /**
   * @param {string} titulo
   * @param {object} opciones {retrasoMs: no aparece si todo termina antes}
   */
  constructor(titulo, opciones) {
    const config = opciones || {};
    this.visible = false;
    this.cerrado = false;
    this.inicio = performance.now();

    this.fondo = document.createElement("div");
    this.fondo.className = "capa-carga";
    this.fondo.setAttribute("role", "status");
    this.fondo.setAttribute("aria-live", "polite");
    this.fondo.innerHTML =
      "<div class='dialogo-carga'>" +
        "<div class='girador'></div>" +
        "<h2 class='titulo-carga'></h2>" +
        "<p class='fase-carga tenue'></p>" +
        "<div class='barra'><span class='relleno-carga' style='width:0%'></span></div>" +
        "<p class='detalle-carga mono tenue'></p>" +
      "</div>";

    this.el = {
      titulo: this.fondo.querySelector(".titulo-carga"),
      fase: this.fondo.querySelector(".fase-carga"),
      relleno: this.fondo.querySelector(".relleno-carga"),
      detalle: this.fondo.querySelector(".detalle-carga"),
    };
    this.el.titulo.textContent = titulo || "Cargando";

    /* Si todo tarda menos que el retraso, no llega a verse: un parpadeo del
       indicador molesta más que su ausencia. */
    const retraso = config.retrasoMs === undefined ? 150 : config.retrasoMs;
    this.temporizador = setTimeout(() => this.mostrar(), retraso);
  }

  mostrar() {
    if (this.cerrado || this.visible) return;
    document.body.appendChild(this.fondo);
    this.visible = true;
  }

  fase(texto) {
    this.el.fase.textContent = texto || "";
  }

  /**
   * @param {number} hechos
   * @param {number} total     0 o negativo => barra indeterminada
   * @param {string} unidad
   */
  progreso(hechos, total, unidad) {
    if (total > 0) {
      const pct = Math.max(0, Math.min(100, (hechos / total) * 100));
      this.el.relleno.style.width = pct + "%";
      this.el.relleno.classList.remove("indeterminada");
      this.el.detalle.textContent =
        hechos + " de " + total + (unidad ? " " + unidad : "") +
        " · " + pct.toFixed(0) + " %";
    } else {
      this.el.relleno.classList.add("indeterminada");
      this.el.relleno.style.width = "100%";
      this.el.detalle.textContent = "";
    }
  }

  error(mensaje) {
    this.mostrar();
    this.el.fase.textContent = mensaje;
    this.el.fase.classList.add("fase-error");
    this.el.relleno.classList.remove("indeterminada");
  }

  cerrar() {
    this.cerrado = true;
    clearTimeout(this.temporizador);
    if (!this.visible) return;
    this.fondo.classList.add("saliendo");
    setTimeout(() => this.fondo.remove(), 180);
    this.visible = false;
  }

  /** Segundos transcurridos desde que se creó. */
  get transcurrido() {
    return (performance.now() - this.inicio) / 1000;
  }
}


/* Aviso efímero en una esquina, para confirmar una acción sin interrumpir. */
class Avisos {
  static contenedor() {
    let caja = document.getElementById("avisos-flotantes");
    if (!caja) {
      caja = document.createElement("div");
      caja.id = "avisos-flotantes";
      caja.className = "avisos-flotantes";
      document.body.appendChild(caja);
    }
    return caja;
  }

  /**
   * @param {string} mensaje  admite HTML ya escapado por quien llama
   * @param {string} clase    exito|error|info
   * @param {number} duracion ms antes de desvanecerse
   */
  static mostrar(mensaje, clase, duracion) {
    const aviso = document.createElement("div");
    aviso.className = "aviso-flotante " + (clase || "info");
    aviso.innerHTML = mensaje;
    Avisos.contenedor().appendChild(aviso);

    /* Reflow para que la transición de entrada se dispare. */
    void aviso.offsetWidth;
    aviso.classList.add("visible");

    const ms = duracion === undefined ? 2600 : duracion;
    setTimeout(() => {
      aviso.classList.remove("visible");
      setTimeout(() => aviso.remove(), 300);
    }, ms);
    return aviso;
  }

  static exito(mensaje, duracion) {
    return Avisos.mostrar(mensaje, "exito", duracion);
  }

  static error(mensaje, duracion) {
    return Avisos.mostrar(mensaje, "error", duracion === undefined ? 4500 : duracion);
  }
}
