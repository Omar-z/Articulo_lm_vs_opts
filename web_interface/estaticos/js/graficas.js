/* Envoltura sobre Chart.js.
   Aísla al resto del código de la API de la librería y degrada con elegancia si
   el archivo vendorizado no está presente. */

class GraficaDeCurvas {
  static PALETA = [
    "#6c9cf7", "#4ec9a0", "#e3b341", "#f2687a",
    "#b98cf0", "#4fc3d9", "#f09a5b", "#8fbf5a",
  ];

  /**
   * @param {HTMLCanvasElement} lienzo
   * @param {object} opciones  {ejeY: 'lineal'|'log', etiquetaX, etiquetaY}
   */
  constructor(lienzo, opciones) {
    this.disponible = typeof window.Chart !== "undefined";
    this.series = new Map();
    if (!this.disponible) return;

    const config = opciones || {};
    const estilo = GraficaDeCurvas.colores();

    this.grafica = new Chart(lienzo.getContext("2d"), {
      type: "line",
      data: { datasets: [] },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        parsing: false,
        normalized: true,
        interaction: { mode: "nearest", axis: "x", intersect: false },
        scales: {
          x: {
            type: "linear",
            title: { display: true, text: config.etiquetaX || "época", color: estilo.tenue },
            ticks: { color: estilo.tenue, maxTicksLimit: 10 },
            grid: { color: estilo.rejilla },
          },
          y: {
            type: config.ejeY === "lineal" ? "linear" : "logarithmic",
            title: { display: true, text: config.etiquetaY || "pérdida", color: estilo.tenue },
            ticks: {
              color: estilo.tenue,
              maxTicksLimit: 8,
              /* Sin esto, la escala logarítmica escribe "0.0000000000010". */
              callback: (valor) => formatearNumero(valor, 3),
            },
            grid: { color: estilo.rejilla },
          },
        },
        plugins: {
          legend: {
            labels: {
              color: estilo.texto,
              boxWidth: 12,
              font: { size: 11 },
              /* Una entrada por optimizador, no por corrida: con 3 optimizadores
                 × 4 reglas × 3 corridas la leyenda taparía la gráfica.

                 `fontColor` es obligatorio aquí: Chart.js pinta el texto con
                 `ctx.fillStyle = legendItem.fontColor`, y `labels.color` solo lo
                 aplica su `generateLabels` por defecto. Al sustituirlo, sin esta
                 propiedad el color queda sin definir y el canvas lo dibuja en
                 negro, ilegible sobre el tema oscuro.

                 Se lee del CSS en cada llamada, no del valor capturado al crear
                 la gráfica, para que siga el tema si el sistema cambia de claro
                 a oscuro con la página abierta. */
              generateLabels: (gr) => {
                const colorTexto = estilo.texto;
                const vistos = new Map();
                gr.data.datasets.forEach((conjunto) => {
                  const nombre = (conjunto.label || "").split(" · ")[0];
                  if (!vistos.has(nombre)) {
                    vistos.set(nombre, {
                      text: nombre,
                      fillStyle: conjunto.borderColor,
                      strokeStyle: conjunto.borderColor,
                      fontColor: colorTexto,
                      lineWidth: 1,
                      hidden: false,
                      datasetIndex: undefined,
                      grupo: nombre,
                    });
                  }
                });
                return Array.from(vistos.values());
              },
            },
            /* Al pulsar una entrada se ocultan/muestran todas las series de ese
               optimizador a la vez. */
            onClick: (evento, elemento, leyenda) => {
              const grafica = leyenda.chart;
              const grupo = elemento.grupo;
              let ocultar = null;
              grafica.data.datasets.forEach((conjunto, indice) => {
                if ((conjunto.label || "").split(" · ")[0] !== grupo) return;
                if (ocultar === null) ocultar = grafica.isDatasetVisible(indice);
                grafica.setDatasetVisibility(indice, !ocultar);
              });
              grafica.update();
            },
          },
          tooltip: {
            callbacks: {
              label: (ctx) => ctx.dataset.label + ": " + formatearNumero(ctx.parsed.y),
            },
          },
        },
      },
    });
  }

  static colores() {
    const raiz = getComputedStyle(document.documentElement);
    return {
      texto: raiz.getPropertyValue("--texto").trim() || "#f4c0f4",//"#e6e8ee",
      tenue: raiz.getPropertyValue("--texto-tenue").trim() || "#9aa1b1",
      rejilla: raiz.getPropertyValue("--borde").trim() || "#2a2f3d",
    };
  }

  /* Color por optimizador, asignado en orden de aparición dentro de esta gráfica.
     Con un hash del nombre, SGD y AdamW caían en el mismo color y la comparación
     resultaba ilegible; por orden no hay colisiones hasta 8 optimizadores. */
  colorDe(grupo) {
    if (!this.colores) this.colores = new Map();
    if (!this.colores.has(grupo)) {
      this.colores.set(grupo, GraficaDeCurvas.PALETA[this.colores.size % GraficaDeCurvas.PALETA.length]);
    }
    return this.colores.get(grupo);
  }

  /** Crea la serie si no existe y devuelve su conjunto de datos. */
  asegurarSerie(clave, etiqueta, destacada) {
    if (!this.disponible) return null;
    if (this.series.has(clave)) return this.series.get(clave);

    const color = this.colorDe(clave.split("|")[0]);
    const conjunto = {
      label: etiqueta || clave,
      data: [],
      borderColor: color,
      backgroundColor: color,
      borderWidth: destacada ? 2 : 1,
      pointRadius: 0,
      tension: 0.1,
      /* Las corridas anteriores se atenúan para que destaque la que está viva. */
      borderDash: destacada ? [] : [3, 3],
      hidden: false,
    };
    this.grafica.data.datasets.push(conjunto);
    this.series.set(clave, conjunto);
    return conjunto;
  }

  agregarPunto(clave, x, y, etiqueta) {
    const serie = this.asegurarSerie(clave, etiqueta, true);
    if (!serie) return;
    serie.data.push({ x: x, y: y });
  }

  fijarSerie(clave, xs, ys, etiqueta, destacada) {
    const serie = this.asegurarSerie(clave, etiqueta, destacada);
    if (!serie) return;
    serie.data = xs.map((x, i) => ({ x: x, y: ys[i] }));
  }

  /** Marca todas las series como históricas menos la indicada. */
  destacar(claveActiva) {
    if (!this.disponible) return;
    this.series.forEach((serie, clave) => {
      const activa = clave === claveActiva;
      serie.borderWidth = activa ? 2 : 1;
      serie.borderDash = activa ? [] : [3, 3];
    });
  }

  /** Cambia el eje Y entre logarítmico y lineal. */
  cambiarEscala(escala) {
    if (!this.disponible) return;
    this.grafica.options.scales.y.type =
      escala === "lineal" ? "linear" : "logarithmic";
    this.grafica.update();
  }

  refrescar() {
    if (this.disponible) this.grafica.update("none");
  }

  limpiar() {
    if (!this.disponible) return;
    this.grafica.data.datasets = [];
    this.series.clear();
    if (this.colores) this.colores.clear();
    this.grafica.update("none");
  }
}
