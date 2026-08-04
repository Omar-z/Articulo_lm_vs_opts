/* Constructor visual de experimentos.

   Arma el mismo objeto que consume la API de trabajos y que se puede descargar
   como JSON para `rpipeline.py`. */

class ConstructorDeExperimento {
  constructor() {
    this.datasets = [];
    this.optimizadores = [];      // catálogo
    this.seleccionados = [];      // {nombre, descriptor, renderizador}

    this.el = {
      dataset: document.getElementById("dataset"),
      selector: document.getElementById("selector-optimizador"),
      lista: document.getElementById("lista-optimizadores"),
      resumen: document.getElementById("resumen-total"),
      validacion: document.getElementById("resultado-validacion"),
      avisoDataset: document.getElementById("aviso-dataset"),
      previa: document.getElementById("previa-dataset"),
      proporciones: document.getElementById("proporciones-efectivas"),
    };

    this.el.dataset.addEventListener("change", () => this.alElegirDataset());
    document.getElementById("btn-agregar-optimizador")
      .addEventListener("click", () => this.agregarOptimizador());
    document.getElementById("btn-validar").addEventListener("click", () => this.validar());
    document.getElementById("btn-ejecutar").addEventListener("click", () => this.ejecutar());
    document.getElementById("btn-descargar").addEventListener("click", () => this.descargar());
    document.getElementById("btn-guardar").addEventListener("click", () => this.guardar());

    document.getElementById("minilotes").addEventListener("change", (e) => {
      document.getElementById("lote_size").disabled = !e.target.checked;
    });

    ["reglas_inicial", "reglas_total", "corridas", "test_size", "val_size"]
      .forEach((id) => document.getElementById(id)
        .addEventListener("input", () => this.actualizarResumen()));

    this.cargarCatalogos();
  }

  /* --- carga inicial ---------------------------------------------------- */

  async cargarCatalogos() {
    const [datasets, optimizadores, perdidas, dispositivos, politicas] =
      await Promise.all([
        ClienteAPI.obtener("/api/datasets").catch(() => []),
        ClienteAPI.obtener("/api/optimizadores").catch(() => ({ optimizadores: [] })),
        ClienteAPI.obtener("/api/funciones-perdida").catch(() => ["SSE"]),
        ClienteAPI.obtener("/api/dispositivos").catch(() => []),
        ClienteAPI.obtener("/api/politicas").catch(() => ({ politicas: [] })),
      ]);

    this.pintarPoliticas(politicas.politicas || []);

    this.datasets = datasets;
    this.el.dataset.innerHTML =
      "<option value=''>— elige un dataset —</option>" +
      datasets.map((d) =>
        "<option value='" + d.id + "'>" + escaparHTML(d.nombre) +
        " (" + d.origen + ")</option>").join("");

    this.optimizadores = optimizadores.optimizadores || [];
    this.el.selector.innerHTML = this.optimizadores.map((o) =>
      "<option value='" + o.nombre + "'>" + escaparHTML(o.nombre) +
      (o.origen !== "torch" ? " · " + o.origen : "") +
      (o.compatible ? "" : " ⚠") + "</option>").join("");

    document.getElementById("funcion_perdida").innerHTML =
      perdidas.map((p) => "<option value='" + p + "'" +
        (p === "SSE" ? " selected" : "") + ">" + p + "</option>").join("");

    document.getElementById("dispositivo").innerHTML = dispositivos.map((d) =>
      "<option value='" + d.id + "'" +
      (d.disponible && d.compatible ? "" : " disabled") + ">" +
      escaparHTML(d.etiqueta) +
      (d.compatible ? "" : " — " + escaparHTML(d.motivo || "no compatible")) +
      "</option>").join("");

    /* Arranque cómodo: LM frente a Adam es la comparación de referencia. Estos no
       avisan, porque no los ha pedido el usuario. */
    ["LM", "Adam"].forEach((n) => {
      if (this.optimizadores.some((o) => o.nombre === n)) {
        this.agregarOptimizador(n, true);
      }
    });

    /* /datasets?dataset=<id> preselecciona */
    const pedido = new URLSearchParams(location.search).get("dataset");
    if (pedido && datasets.some((d) => d.id === pedido)) {
      this.el.dataset.value = pedido;
      await this.alElegirDataset();
    }
    this.actualizarResumen();
  }

  /* --- políticas de paro ------------------------------------------------- */

  pintarPoliticas(politicas) {
    const caja = document.getElementById("lista-politicas-constructor");
    if (!caja) return;
    if (!politicas.length) {
      caja.innerHTML = "<p class='tenue'>No hay políticas disponibles.</p>";
      return;
    }
    /* Las del núcleo van marcadas por defecto: es el comportamiento del CLI. */
    caja.innerHTML = politicas.map((p) =>{
      const insignia = p.origen === "nucleo" ? "pendiente" : "terminado";
      return "<label class='casilla'>" +
      "<input type='checkbox' data-politica='" + escaparHTML(p.nombre_clase) + "'" +
      (p.origen === "nucleo" ? " checked" : "") + ">"+
      "<span>" + escaparHTML(p.etiqueta) + " "+
      "<span class='insignia "+insignia+"'>" + (p.origen === "nucleo" ? "proyecto" : "custom") +
      "</span></span></label>"
    }).join("");
  }

  politicasElegidas() {
    return [...document.querySelectorAll("[data-politica]:checked")]
      .map((c) => c.dataset.politica);
  }

  /* --- dataset ---------------------------------------------------------- */

  async alElegirDataset() {
    const id = this.el.dataset.value;
    this.el.avisoDataset.style.display = "none";
    this.el.previa.innerHTML = "";
    if (!id) return;

    try {
      const esquema = await ClienteAPI.obtener(
        "/api/datasets/" + id + "/esquema-sugerido");
      this.esquema = esquema;

      document.getElementById("tipo").value = esquema.tipo;
      document.getElementById("dataset_entradas").value = esquema.dataset_entradas;
      document.getElementById("dataset_salidas").value = esquema.dataset_salidas;
      document.getElementById("dataset_target_col").value =
        esquema.dataset_target_col === null ? "" : esquema.dataset_target_col;
      document.getElementById("dataset_map_col").value =
        esquema.dataset_map_col ? JSON.stringify(esquema.dataset_map_col) : "";
      document.getElementById("dataset_sep").value =
        esquema.sep && esquema.sep !== "," ? esquema.sep : "";

      this.el.avisoDataset.innerHTML =
        "<strong>Configuración deducida del archivo.</strong> " +
        escaparHTML(esquema.aviso || "") +
        " Revísala y corrige lo que haga falta.";
      this.el.avisoDataset.style.display = "block";

      await this.cargarPrevia(id);
    } catch (error) {
      this.el.avisoDataset.className = "aviso error";
      this.el.avisoDataset.textContent = "No se pudo leer el dataset: " + error.message;
      this.el.avisoDataset.style.display = "block";
    }
    this.actualizarResumen();
  }

  async cargarPrevia(id) {
    try {
      const datos = await ClienteAPI.obtener(
        "/api/datasets/" + id + "/vista-previa?filas=5");
      this.el.previa.innerHTML =
        "<table><thead><tr>" +
        datos.columnas.map((c, i) => "<th class='num'>" + escaparHTML(c) +
          "<br><span class='tenue'>col " + i + "</span></th>").join("") +
        "</tr></thead><tbody>" +
        datos.filas.map((f) => "<tr>" + f.map((v) =>
          "<td class='num'>" + (typeof v === "number" ? formatearNumero(v, 3) :
            escaparHTML(v)) + "</td>").join("") + "</tr>").join("") +
        "</tbody></table>";
    } catch (error) { /* la previa es opcional */ }
  }

  /* --- optimizadores ---------------------------------------------------- */

  agregarOptimizador(nombreForzado, silencioso) {
    const nombre = nombreForzado || this.el.selector.value;
    if (!nombre) return;
    if (this.seleccionados.some((s) => s.nombre === nombre)) {
      Avisos.error("<strong>" + escaparHTML(nombre) +
        "</strong> ya está en el experimento.");
      return;
    }
    const descriptor = this.optimizadores.find((o) => o.nombre === nombre);
    if (!descriptor) return;

    if (!descriptor.compatible &&
        !confirm(descriptor.nombre + " no es compatible con este modelo:\n\n" +
                 descriptor.motivo_incompatible + "\n\n¿Añadirlo de todas formas?")) {
      return;
    }

    const tarjeta = document.createElement("div");
    tarjeta.className = "tarjeta-optimizador";
    tarjeta.innerHTML =
      "<div class='cabecera-optimizador'>" +
        "<div><strong>" + escaparHTML(descriptor.nombre) + "</strong>" +
        "<span class='tenue'> · " + descriptor.origen + "</span>" +
        (descriptor.doc ? "<p class='tenue doc'>" + escaparHTML(descriptor.doc) + "</p>" : "") +
        "</div>" +
        "<button class='peligro' data-quitar>Quitar</button>" +
      "</div><div class='cuerpo-optimizador'></div>";

    if (this.el.lista.querySelector(".vacio")) this.el.lista.innerHTML = "";
    this.el.lista.appendChild(tarjeta);

    const renderizador = new RenderizadorDeHiperparametros(
      tarjeta.querySelector(".cuerpo-optimizador"), descriptor, {});

    const entrada = { nombre, descriptor, renderizador, tarjeta };
    this.seleccionados.push(entrada);

    tarjeta.querySelector("[data-quitar]").addEventListener("click", () => {
      this.seleccionados = this.seleccionados.filter((s) => s !== entrada);
      tarjeta.remove();
      if (!this.seleccionados.length) {
        this.el.lista.innerHTML = "<p class='vacio'>Añade al menos un optimizador.</p>";
      }
      Avisos.mostrar("<strong>" + escaparHTML(nombre) +
        "</strong> se quitó del experimento.", "info");
      this.actualizarResumen();
    });

    this.actualizarResumen();

    if (!silencioso) {
      /* Confirmación efímera: la tarjeta puede quedar fuera de la vista y sin
         esto no queda claro que el botón haya hecho algo. */
      Avisos.exito(
        "<strong>" + escaparHTML(nombre) + "</strong> añadido al experimento" +
        " · ahora son " + this.seleccionados.length + " optimizador(es)");
      tarjeta.classList.add("recien-agregada");
      tarjeta.scrollIntoView({ behavior: "smooth", block: "nearest" });
      setTimeout(() => tarjeta.classList.remove("recien-agregada"), 1400);
    }
  }

  /* --- construcción del objeto ------------------------------------------ */

  numero(id) {
    const valor = document.getElementById(id).value;
    return valor === "" ? null : Number(valor);
  }

  configuracion() {
    const dataset = this.datasets.find((d) => d.id === this.el.dataset.value);
    if (!dataset) throw new Error("Elige un dataset");
    if (!this.seleccionados.length) throw new Error("Añade al menos un optimizador");

    let mapa = null;
    const textoMapa = document.getElementById("dataset_map_col").value.trim();
    if (textoMapa) {
      try { mapa = JSON.parse(textoMapa); }
      catch (e) { throw new Error("El mapa de clases no es un JSON válido: " + e.message); }
    }

    /* Un par de .mat se envía como lista de dos rutas. */
    const ruta = (this.esquema && Array.isArray(this.esquema.dataset_path))
      ? this.esquema.dataset_path
      : dataset.ruta_relativa;

    const separador = document.getElementById("dataset_sep").value;
    const minilotes = document.getElementById("minilotes").checked;

    return {
      optimizadores: this.seleccionados.map((s) => ({
        nombre: s.nombre,
        params: s.renderizador.recolectar(),
      })),
      experimentacion: {
        dataset_path: ruta,
        dataset_header: (this.esquema && this.esquema.header === 0) ? 0 : null,
        dataset_sep: separador || null,
        dataset_target_col: this.numero("dataset_target_col"),
        dataset_map_col: mapa,
        dataset_entradas: this.numero("dataset_entradas"),
        dataset_salidas: this.numero("dataset_salidas"),
        corridas: this.numero("corridas"),
        epocas: this.numero("epocas"),
        tolerancia: Number(document.getElementById("tolerancia").value) || 1e-12,
        fallos: this.numero("fallos"),
        funcion_perdida: document.getElementById("funcion_perdida").value,
        tipo: document.getElementById("tipo").value,
        reglas_inicial: this.numero("reglas_inicial"),
        reglas_total: this.numero("reglas_total"),
        train_size: 0.6,
        test_size: this.numero("test_size"),
        val_size: this.numero("val_size"),
        lote_size: minilotes ? this.numero("lote_size") : null,
      },
    };
  }

  solicitud() {
    return {
      tipo: "experimento",
      config: this.configuracion(),
      dispositivo: document.getElementById("dispositivo").value || "cpu",
      minilotes: document.getElementById("minilotes").checked,
      semilla_maestra: this.numero("semilla"),
      politicas: this.politicasElegidas(),
    };
  }

  actualizarResumen() {
    const reglas = Math.max(0,
      (this.numero("reglas_total") || 0) - (this.numero("reglas_inicial") || 0) + 1);
    const total = reglas * (this.numero("corridas") || 0) * this.seleccionados.length;
    const epocas = this.numero("epocas") || 0;

    this.el.resumen.innerHTML =
      "<strong>" + total + "</strong> entrenamiento(s) de hasta " + epocas +
      " épocas: " + reglas + " configuración(es) de reglas × " +
      this.seleccionados.length + " optimizador(es) × " +
      (this.numero("corridas") || 0) + " corrida(s)." +
      (total > 200 ? " <span class='alerta'>Esto puede tardar horas.</span>" : "");

    /* `train_size` no interviene: el reparto sale de test_size y val_size. */
    const prueba = this.numero("test_size") || 0;
    const val = this.numero("val_size") || 0;
    const pct = (x) => (x * 100).toFixed(0) + " %";
    this.el.proporciones.textContent =
      "Reparto efectivo: " + pct(1 - prueba) + " entrenamiento · " +
      pct(prueba * (1 - val)) + " prueba · " + pct(prueba * val) + " validación " +
      "(igual que el script: aparta la prueba y luego subdivide).";
  }

  /* --- acciones --------------------------------------------------------- */

  mostrar(html, clase) {
    this.el.validacion.innerHTML =
      "<div class='aviso " + (clase || "info") + "'>" + html + "</div>";
  }

  async validar() {
    let config;
    try { config = this.configuracion(); }
    catch (e) { this.mostrar(escaparHTML(e.message), "error"); return null; }

    try {
      const r = await ClienteAPI.enviar("/api/trabajos/validar", config);
      let html = r.valido
        ? "<strong>Configuración válida.</strong> " + r.resumen.total_entrenamientos +
          " entrenamientos."
        : "<strong>Hay errores:</strong><ul>" +
          r.errores.map((e) => "<li>" + escaparHTML(e.campo) + ": " +
            escaparHTML(e.mensaje) + "</li>").join("") + "</ul>";
      if (r.avisos.length) {
        html += "<ul class='tenue'>" +
          r.avisos.map((a) => "<li>" + escaparHTML(a) + "</li>").join("") + "</ul>";
      }
      this.mostrar(html, r.valido ? "info" : "error");
      return r.valido ? config : null;
    } catch (error) {
      this.mostrar(escaparHTML(error.message), "error");
      return null;
    }
  }

  async ejecutar() {
    const config = await this.validar();
    if (!config) return;
    try {
      const respuesta = await ClienteAPI.enviar("/api/trabajos", this.solicitud());
      location.href = "/monitor/" + respuesta.id_trabajo;
    } catch (error) {
      this.mostrar("No se pudo lanzar: " + escaparHTML(error.message), "error");
    }
  }

  async descargar() {
    let config;
    try { config = this.configuracion(); }
    catch (e) { this.mostrar(escaparHTML(e.message), "error"); return; }

    try {
      const crudo = await ClienteAPI.enviar("/api/experimentos/exportar", config);
      const blob = new Blob([JSON.stringify(crudo, null, 2)], { type: "application/json" });
      const enlace = document.createElement("a");
      enlace.href = URL.createObjectURL(blob);
      enlace.download = "experimento_tests.json";
      enlace.click();
      URL.revokeObjectURL(enlace.href);
      const salida = escaparHTML(crudo.experimentacion.resultados_path);
      this.mostrar(
        "JSON descargado. Ejecútalo con " +
        "<code>python rpipeline.py experimento_tests.json cpu</code>.<br>" +
        "<strong>Ojo:</strong> el script escribirá en <code>" + salida + "</code> y " +
        "<strong>sobrescribirá</strong> el <code>resultados.csv</code> y el " +
        "<code>resultados.json</code> que ya haya ahí. Cambia " +
        "<code>resultados_path</code> en el archivo si quieres conservarlos.",
        "info");
    } catch (error) {
      this.mostrar("No se pudo exportar: " + escaparHTML(error.message), "error");
    }
  }

  async guardar() {
    let config;
    try { config = this.configuracion(); }
    catch (e) { this.mostrar(escaparHTML(e.message), "error"); return; }

    const nombre = prompt("Nombre para esta configuración:", "mi experimento");
    if (nombre === null) return;
    try {
      const r = await ClienteAPI.enviar(
        "/api/experimentos?nombre=" + encodeURIComponent(nombre), config);
      this.mostrar("Guardada como <code>" + escaparHTML(r.id) + "</code>.", "info");
    } catch (error) {
      this.mostrar("No se pudo guardar: " + escaparHTML(error.message), "error");
    }
  }
}

document.addEventListener("DOMContentLoaded", () => new ConstructorDeExperimento());
