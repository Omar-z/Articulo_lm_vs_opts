/* Explorador de datasets: catálogo y subida. */

class ExploradorDeDatasets {
  constructor() {
    this.cuerpo = document.querySelector("#tabla-datasets tbody");
    this.estado = document.getElementById("estado-subida");

    document.getElementById("filtro-origen")
      .addEventListener("change", (e) => this.cargar(e.target.value));

    const entrada = document.getElementById("archivo");
    const zona = document.getElementById("zona-subida");
    entrada.addEventListener("change", () => {
      if (entrada.files.length) this.subir(entrada.files[0]);
    });

    ["dragenter", "dragover"].forEach((evento) =>
      zona.addEventListener(evento, (e) => {
        e.preventDefault();
        zona.classList.add("activa");
      })
    );
    ["dragleave", "drop"].forEach((evento) =>
      zona.addEventListener(evento, (e) => {
        e.preventDefault();
        zona.classList.remove("activa");
      })
    );
    zona.addEventListener("drop", (e) => {
      if (e.dataTransfer.files.length) this.subir(e.dataTransfer.files[0]);
    });

    this.cargar("");
  }

  async cargar(origen) {
    try {
      this.pintar(await ClienteAPI.obtener(
        "/api/datasets" + (origen ? "?origen=" + origen : "")));
    } catch (error) {
      this.cuerpo.innerHTML =
        "<tr><td colspan='5' class='vacio'>No se pudo cargar: " +
        escaparHTML(error.message) + "</td></tr>";
    }
  }

  pintar(datasets) {
    if (!datasets.length) {
      this.cuerpo.innerHTML =
        "<tr><td colspan='5' class='vacio'>No hay datasets.</td></tr>";
      return;
    }
    this.cuerpo.innerHTML = datasets.map((d) => {
      const kb = d.bytes / 1024;
      const tamano = kb > 1024 ? (kb / 1024).toFixed(1) + " MB" : kb.toFixed(0) + " KB";
      const borrar = d.editable
        ? " <button class='peligro' data-borrar='" + d.id + "'>Borrar</button>" : "";
      return "<tr>" +
        "<td><a href='/datasets/" + d.id + "'>" + escaparHTML(d.nombre) + "</a></td>" +
        "<td class='tenue mono'>" + escaparHTML(d.ruta_relativa) + "</td>" +
        "<td><span class='insignia " + (d.origen === 'web' ? 'ejecutando' : 'pendiente') +
          "'>" + (d.origen === "web" ? "subido" : "repositorio") + "</span></td>" +
        "<td class='num'>" + tamano + "</td>" +
        "<td><a class='boton' href='/constructor?dataset=" + d.id + "'>Usar</a>" +
          borrar + "</td></tr>";
    }).join("");

    this.cuerpo.querySelectorAll("[data-borrar]").forEach((boton) => {
      boton.addEventListener("click", async () => {
        if (!confirm("¿Borrar este dataset subido?")) return;
        try {
          await ClienteAPI.borrar("/api/datasets/" + boton.dataset.borrar);
          this.cargar(document.getElementById("filtro-origen").value);
        } catch (error) {
          this.estado.textContent = "No se pudo borrar: " + error.message;
        }
      });
    });
  }

  async subir(archivo) {
    this.estado.textContent = "Subiendo " + archivo.name + "…";
    const formulario = new FormData();
    formulario.append("archivo", archivo);
    try {
      const dataset = await ClienteAPI.enviar("/api/datasets", formulario);
      this.estado.textContent = "Listo: " + dataset.nombre;
      this.cargar(document.getElementById("filtro-origen").value);
    } catch (error) {
      this.estado.textContent = "No se pudo subir: " + error.message;
    }
  }
}

document.addEventListener("DOMContentLoaded", () => new ExploradorDeDatasets());
