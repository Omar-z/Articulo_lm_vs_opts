/* Editor y catálogo de políticas de paro.

   El nombre que escribe el usuario se convierte en el nombre de la clase, y el
   editor se prellena con el esqueleto correspondiente. Mientras no se toque el
   código a mano, la plantilla se regenera al cambiar el nombre. */

class EditorDePoliticas {
  constructor() {
    this.codigoTocado = false;
    this.editando = null;   // nombre de clase que se está reemplazando

    this.el = {
      nombre: document.getElementById("nombre-politica"),
      codigo: document.getElementById("codigo-politica"),
      pista: document.getElementById("pista-clase"),
      resultado: document.getElementById("resultado-politica"),
      lista: document.getElementById("lista-politicas"),
      cuenta: document.getElementById("cuenta-politicas"),
      archivo: document.getElementById("archivo-politicas"),
      ruta: document.getElementById("ruta-archivo"),
      avisoSeguridad: document.getElementById("aviso-seguridad"),
      guardar: document.getElementById("btn-guardar"),
    };

    let temporizador = null;
    this.el.nombre.addEventListener("input", () => {
      clearTimeout(temporizador);
      temporizador = setTimeout(() => this.actualizarPlantilla(), 250);
    });

    /* En cuanto el usuario escribe en el editor, deja de regenerarse solo: sería
       muy molesto perder lo escrito por corregir una letra del nombre. */
    this.el.codigo.addEventListener("input", () => { this.codigoTocado = true; });

    /* Tab inserta indentación en vez de saltar de campo. */
    this.el.codigo.addEventListener("keydown", (e) => this.tabulacion(e));

    document.getElementById("btn-validar").addEventListener("click", () => this.validar());
    this.el.guardar.addEventListener("click", () => this.guardar());
    document.getElementById("btn-limpiar").addEventListener("click", () => this.limpiar());
    document.getElementById("btn-recargar").addEventListener("click", () => this.recargar());

    this.actualizarPlantilla();
    this.cargar();
  }

  /* --- editor ----------------------------------------------------------- */

  tabulacion(evento) {
    if (evento.key !== "Tab") return;
    evento.preventDefault();
    const campo = evento.target;
    const inicio = campo.selectionStart;
    const fin = campo.selectionEnd;
    campo.value = campo.value.slice(0, inicio) + "    " + campo.value.slice(fin);
    campo.selectionStart = campo.selectionEnd = inicio + 4;
  }

  async actualizarPlantilla() {
    const nombre = this.el.nombre.value;
    try {
      const datos = await ClienteAPI.obtener(
        "/api/politicas/plantilla?nombre=" + encodeURIComponent(nombre));
      this.el.pista.innerHTML =
        "La clase se llamará <code>" + escaparHTML(datos.nombre_clase) + "</code>.";
      if (!this.codigoTocado) this.el.codigo.value = datos.codigo;
    } catch (error) {
      this.el.pista.innerHTML =
        "<span class='alerta'>" + escaparHTML(error.message) + "</span>";
    }
  }

  limpiar() {
    this.codigoTocado = false;
    this.editando = null;
    this.el.nombre.value = "";
    this.el.resultado.innerHTML = "";
    this.el.guardar.textContent = "Guardar política";
    this.actualizarPlantilla();
    this.el.nombre.focus();
  }

  mostrar(html, clase) {
    this.el.resultado.innerHTML =
      "<div class='aviso " + (clase || "info") + "'>" + html + "</div>";
  }

  cuerpo() {
    return {
      nombre: this.el.nombre.value || "Mi política",
      codigo: this.el.codigo.value,
      nombre_clase: this.editando || undefined,
    };
  }

  /* --- acciones --------------------------------------------------------- */

  async validar(silencioso) {
    try {
      const r = await ClienteAPI.enviar("/api/politicas/validar", this.cuerpo());
      if (!r.valido) {
        this.mostrar(
          "<strong>La política tiene problemas:</strong><ul>" +
          r.errores.map((e) => "<li>" + escaparHTML(e) + "</li>").join("") +
          "</ul>", "error");
        return false;
      }

      if (!silencioso) {
        /* Se enseña qué contestó la política a una secuencia de pérdidas, que es
           la forma más rápida de ver si la condición está al revés. */
        const perdidas = [100, 10, 1, 0.1, 0.01, 1e-9, 0];
        const tabla = r.prueba.resultados
          .map((v, i) => "<code>" + perdidas[i] + "</code> → " +
            (v ? "<strong>detiene</strong>" : "sigue")).join(" · ");
        let html = "<strong>Válida.</strong> Se llamará <code>" +
          escaparHTML(r.prueba.nombre) + "</code>.<br>" +
          "<span class='tenue'>Prueba con pérdidas: " + tabla + "</span>";
        if (r.avisos.length) {
          html += "<ul class='tenue'>" +
            r.avisos.map((a) => "<li>" + escaparHTML(a) + "</li>").join("") + "</ul>";
        }
        this.mostrar(html, "info");
      }
      return true;
    } catch (error) {
      this.mostrar(escaparHTML(error.message), "error");
      return false;
    }
  }

  async guardar() {
    if (!(await this.validar(true))) return;
    try {
      const politica = await ClienteAPI.enviar("/api/politicas", this.cuerpo());
      Avisos.exito("<strong>" + escaparHTML(politica.nombre_clase) +
        "</strong> guardada y lista para usar.");
      this.mostrar("Guardada como <code>" + escaparHTML(politica.nombre_clase) +
        "</code>. Ya puedes elegirla en el constructor.", "info");
      this.editando = null;
      this.el.guardar.textContent = "Guardar política";
      this.cargar();
    } catch (error) {
      this.mostrar(escaparHTML(error.message), "error");
    }
  }

  async recargar() {
    try {
      const r = await ClienteAPI.enviar("/api/politicas/recargar");
      if (r.error) {
        Avisos.error("El archivo tiene un error: " + escaparHTML(r.error));
      } else {
        Avisos.exito("Recargado: " + (r.cargadas.length || 0) + " política(s) tuya(s).");
      }
      this.cargar();
    } catch (error) {
      Avisos.error(escaparHTML(error.message));
    }
  }

  editar(politica) {
    this.el.nombre.value = politica.etiqueta;
    this.el.codigo.value = politica.codigo;
    this.codigoTocado = true;
    this.editando = politica.nombre_clase;
    this.el.guardar.textContent = "Guardar cambios";
    this.el.pista.innerHTML =
      "Editando <code>" + escaparHTML(politica.nombre_clase) + "</code>.";
    this.el.resultado.innerHTML = "";
    this.el.nombre.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  async borrar(politica) {
    if (!confirm("¿Borrar la política '" + politica.nombre_clase + "' del archivo?")) return;
    try {
      await ClienteAPI.borrar("/api/politicas/" + encodeURIComponent(politica.nombre_clase));
      Avisos.mostrar("<strong>" + escaparHTML(politica.nombre_clase) +
        "</strong> borrada.", "info");
      this.cargar();
    } catch (error) {
      Avisos.error(escaparHTML(error.message));
    }
  }

  /* --- catálogo --------------------------------------------------------- */

  async cargar() {
    try {
      const datos = await ClienteAPI.obtener("/api/politicas");
      this.pintar(datos);
      this.el.ruta.textContent = datos.archivo;

      if (!datos.escritura_permitida) {
        this.el.avisoSeguridad.className = "aviso error";
        this.el.avisoSeguridad.innerHTML =
          "La edición está <strong>desactivada</strong> porque el servidor no es " +
          "local. Arráncalo sin <code>--exponer</code> para poder escribir políticas.";
        ["btn-validar", "btn-guardar"].forEach((id) => {
          document.getElementById(id).disabled = true;
        });
      }

      const archivo = await fetch("/api/politicas/archivo").then((r) => r.text());
      this.el.archivo.textContent = archivo || "(el archivo está vacío)";
    } catch (error) {
      this.el.lista.innerHTML =
        "<p class='vacio'>No se pudo cargar: " + escaparHTML(error.message) + "</p>";
    }
  }

  pintar(datos) {
    const politicas = datos.politicas || [];
    const mias = politicas.filter((p) => p.origen === "usuario").length;
    this.el.cuenta.textContent =
      "(" + politicas.length + ", " + mias + " tuya" + (mias === 1 ? "" : "s") + ")";

    if (datos.error_de_carga) {
      this.el.lista.innerHTML =
        "<div class='aviso error'><strong>El archivo de políticas tiene un error " +
        "y no se pudo cargar:</strong><br><code>" +
        escaparHTML(datos.error_de_carga) + "</code></div>";
      return;
    }

    this.el.lista.innerHTML = politicas.map((p, i) => {
      const insignia = p.origen === "nucleo" ? "pendiente" : "terminado";
      const etiqueta = p.origen === "nucleo" ? "del proyecto" : "tuya";
      const params = p.parametros.length
        ? "<ul class='hiperlista'>" + p.parametros.map((par) =>
            "<li><code>" + escaparHTML(par.nombre) + "</code>" +
            (par.por_defecto === null ? "" : " = " + escaparHTML(String(par.por_defecto))) +
            "</li>").join("") + "</ul>"
        : "";
      const acciones = p.editable
        ? "<div class='acciones' style='margin-top:.6rem'>" +
          "<button data-editar='" + i + "'>Editar</button>" +
          "<button class='peligro' data-borrar='" + i + "'>Borrar</button></div>"
        : "";
      return "<article class='tarjeta-catalogo'>" +
        "<header><strong>" + escaparHTML(p.etiqueta) + "</strong>" +
        "<span class='insignia " + insignia + "'>" + etiqueta + "</span></header>" +
        "<p class='tenue mono' style='font-size:.78rem;margin:.1rem 0 .3rem'>" +
          escaparHTML(p.nombre_clase) + "</p>" +
        (p.doc ? "<p class='tenue doc'>" + escaparHTML(p.doc) + "</p>" : "") +
        params + acciones + "</article>";
    }).join("");

    this.el.lista.querySelectorAll("[data-editar]").forEach((boton) => {
      boton.addEventListener("click", () =>
        this.editar(politicas[Number(boton.dataset.editar)]));
    });
    this.el.lista.querySelectorAll("[data-borrar]").forEach((boton) => {
      boton.addEventListener("click", () =>
        this.borrar(politicas[Number(boton.dataset.borrar)]));
    });
  }
}

document.addEventListener("DOMContentLoaded", () => new EditorDePoliticas());
