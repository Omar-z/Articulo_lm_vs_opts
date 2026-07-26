/* Cliente HTTP de la interfaz.
   Centraliza fetch, cabeceras y el formato de error de FastAPI (campo `detail`). */

const ClienteAPI = (function () {
  async function peticion(metodo, ruta, cuerpo, opciones) {
    const config = {
      method: metodo,
      headers: {},
      ...(opciones || {}),
    };

    if (cuerpo !== undefined && cuerpo !== null) {
      if (cuerpo instanceof FormData) {
        config.body = cuerpo;
      } else {
        config.headers["Content-Type"] = "application/json";
        config.body = JSON.stringify(cuerpo);
      }
    }

    const respuesta = await fetch(ruta, config);

    if (respuesta.status === 204) return null;

    const tipo = respuesta.headers.get("content-type") || "";
    const datos = tipo.includes("application/json")
      ? await respuesta.json()
      : await respuesta.text();

    if (!respuesta.ok) {
      const mensaje =
        (datos && datos.detail) ||
        (typeof datos === "string" && datos) ||
        respuesta.statusText;
      const error = new Error(mensaje);
      error.estado = respuesta.status;
      error.datos = datos;
      throw error;
    }
    return datos;
  }

  return {
    obtener: (ruta, opciones) => peticion("GET", ruta, null, opciones),
    enviar: (ruta, cuerpo) => peticion("POST", ruta, cuerpo),
    actualizar: (ruta, cuerpo) => peticion("PUT", ruta, cuerpo),
    borrar: (ruta) => peticion("DELETE", ruta),

    /** URL de WebSocket para la ruta dada, respetando http/https. */
    urlWebSocket(ruta) {
      const protocolo = location.protocol === "https:" ? "wss:" : "ws:";
      return protocolo + "//" + location.host + ruta;
    },
  };
})();

/** Formatea un número para las tablas: notación científica si es muy pequeño o grande. */
function formatearNumero(valor, decimales) {
  if (valor === null || valor === undefined || Number.isNaN(valor)) return "—";
  if (!Number.isFinite(valor)) return valor > 0 ? "∞" : "−∞";
  const abs = Math.abs(valor);
  if (abs !== 0 && (abs < 1e-4 || abs >= 1e6)) return valor.toExponential(3);
  return valor.toFixed(decimales === undefined ? 6 : decimales);
}

/** Escapa texto antes de insertarlo como HTML. */
function escaparHTML(texto) {
  const div = document.createElement("div");
  div.textContent = texto === null || texto === undefined ? "" : String(texto);
  return div.innerHTML;
}
