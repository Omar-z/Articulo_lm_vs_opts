/* Formulario dinámico de hiperparámetros.

   Consume un DescriptorOptimizador (deducido en el servidor por introspección de
   la firma del constructor) y monta los controles que correspondan. Así el
   formulario no hay que mantenerlo a mano cuando cambia la versión de PyTorch.

   Cada campo lleva una casilla "usar valor por defecto", marcada de inicio. Solo
   los campos desmarcados viajan en `params`, de modo que se envía el diccionario
   mínimo: pasar explícitamente `fused=None` o un `Adafactor.eps` a medio rellenar
   rompería el constructor. */

class RenderizadorDeHiperparametros {
  /**
   * @param {HTMLElement} contenedor
   * @param {object} descriptor  DescriptorOptimizador que devuelve la API
   * @param {object} valores     valores ya fijados (al clonar un experimento)
   */
  constructor(contenedor, descriptor, valores) {
    this.contenedor = contenedor;
    this.descriptor = descriptor;
    this.valores = valores || {};
    this.render();
  }

  render() {
    const principales = this.descriptor.hiperparametros.filter((h) => h.principal);
    const avanzados = this.descriptor.hiperparametros.filter((h) => !h.principal);

    let html = "";
    if (!this.descriptor.hiperparametros.length) {
      html += "<p class='tenue'>Este optimizador no admite hiperparámetros.</p>";
    }
    if (principales.length) {
      html += "<div class='rejilla-campos'>" +
        principales.map((h) => this.campo(h)).join("") + "</div>";
    }
    if (avanzados.length) {
      html += "<details class='avanzados'><summary>Avanzados (" +
        avanzados.length + ")</summary><div class='rejilla-campos'>" +
        avanzados.map((h) => this.campo(h)).join("") + "</div></details>";
    }
    if (!this.descriptor.compatible) {
      html = "<div class='aviso error'>" +
        escaparHTML(this.descriptor.motivo_incompatible) + "</div>" + html;
    }
    this.contenedor.innerHTML = html;
    this.enlazar();
  }

  campo(hiper) {
    const fijado = Object.prototype.hasOwnProperty.call(this.valores, hiper.nombre);
    const valor = fijado ? this.valores[hiper.nombre] : hiper.por_defecto;
    const id = "hp-" + this.descriptor.nombre + "-" + hiper.nombre;

    return "<div class='campo' data-hiper='" + hiper.nombre +
      "' data-tipo='" + hiper.tipo + "' data-aridad='" + hiper.aridad + "'>" +
      "<div class='campo-cabecera'>" +
        "<label for='" + id + "'>" + escaparHTML(hiper.nombre) + "</label>" +
        (hiper.requerido ? "<span class='requerido'>obligatorio</span>" :
          "<label class='usar-defecto'><input type='checkbox' data-defecto " +
          (fijado ? "" : "checked") + "> por defecto</label>") +
      "</div>" +
      this.control(hiper, id, valor, !fijado && !hiper.requerido) +
      "</div>";
  }

  control(hiper, id, valor, deshabilitado) {
    const off = deshabilitado ? " disabled" : "";

    switch (hiper.tipo) {
      case "bool":
        return "<input type='checkbox' id='" + id + "' data-valor" +
          (valor ? " checked" : "") + off + ">";

      case "bool_opcional":
        return "<select id='" + id + "' data-valor" + off + ">" +
          ["auto", "sí", "no"].map((o) => {
            const v = o === "auto" ? "" : (o === "sí" ? "true" : "false");
            const sel = String(valor) === v || (valor === null && v === "") ? " selected" : "";
            return "<option value='" + v + "'" + sel + ">" + o + "</option>";
          }).join("") + "</select>";

      case "tupla": {
        const partes = Array.isArray(valor) ? valor : new Array(hiper.aridad).fill("");
        return "<div class='tupla'>" +
          partes.map((v, i) =>
            "<input type='number' step='any' data-valor data-indice='" + i +
            "' value='" + (v === null || v === undefined ? "" : v) + "'" + off + ">"
          ).join("") + "</div>";
      }

      case "opcion":
        return "<select id='" + id + "' data-valor" + off + ">" +
          (hiper.opciones || []).map((o) =>
            "<option value='" + o + "'" + (o === valor ? " selected" : "") + ">" +
            o + "</option>").join("") + "</select>";

      case "int":
        return "<input type='number' step='1' id='" + id + "' data-valor value='" +
          (valor === null || valor === undefined ? "" : valor) + "'" + off + ">";

      case "texto":
        return "<input type='text' id='" + id + "' data-valor value='" +
          (valor === null || valor === undefined ? "" : escaparHTML(valor)) + "'" + off + ">";

      default:
        return "<input type='number' step='any' id='" + id + "' data-valor value='" +
          (valor === null || valor === undefined ? "" : valor) + "'" + off + ">";
    }
  }

  enlazar() {
    this.contenedor.querySelectorAll("[data-defecto]").forEach((casilla) => {
      casilla.addEventListener("change", () => {
        const campo = casilla.closest(".campo");
        campo.querySelectorAll("[data-valor]").forEach((control) => {
          control.disabled = casilla.checked;
        });
      });
    });
  }

  /** Diccionario con solo los campos que el usuario ha fijado explícitamente. */
  recolectar() {
    const params = {};
    this.contenedor.querySelectorAll(".campo").forEach((campo) => {
      const casilla = campo.querySelector("[data-defecto]");
      if (casilla && casilla.checked) return;  // se deja el valor por defecto

      const nombre = campo.dataset.hiper;
      const tipo = campo.dataset.tipo;
      const controles = campo.querySelectorAll("[data-valor]");
      if (!controles.length) return;

      if (tipo === "tupla") {
        const partes = [...controles].map((c) => (c.value === "" ? null : Number(c.value)));
        /* Una tupla a medio rellenar rompería el constructor: se omite entera. */
        if (partes.every((p) => p === null)) return;
        params[nombre] = partes;
        return;
      }

      const control = controles[0];
      if (tipo === "bool") {
        params[nombre] = control.checked;
      } else if (tipo === "bool_opcional") {
        if (control.value === "") return;   // "auto" equivale a no enviarlo
        params[nombre] = control.value === "true";
      } else if (tipo === "int" || tipo === "float") {
        if (control.value === "") return;
        params[nombre] = Number(control.value);
      } else {
        if (control.value === "") return;
        params[nombre] = control.value;
      }
    });
    return params;
  }
}
