import numpy as np
import matplotlib.pyplot as plt


def graficar_trayectoria(problema, acciones, guardar=None):
    """
    Reconstruye la trayectoria a partir de las acciones de A* y grafica
    el valor del portafolio, el efectivo y el valor invertido por día.
    """

    estado = problema.estado_inicial

    dias = []
    valores = []
    efectivos = []

    for accion in acciones:
        t, efectivo, _ = estado

        # Estado al cierre del día t, después de operar.
        siguiente = problema.transicion(estado, accion)
        _, efectivo_sig, cantidades_sig = siguiente

        valor = efectivo_sig + np.dot(cantidades_sig, problema.precios[t])

        dias.append(t)
        valores.append(valor)
        efectivos.append(efectivo_sig)

        estado = siguiente

    dias = np.array(dias)
    valores = np.array(valores)
    efectivos = np.array(efectivos)

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(dias, valores, label="Valor del portafolio", linewidth=2)
    ax.plot(dias, efectivos, label="Efectivo", linewidth=2)
    ax.plot(dias, valores - efectivos, label="Valor invertido",
            linewidth=1.5, linestyle="--")
    ax.axhline(problema.estado_inicial[1], color="gray",
               linestyle=":", linewidth=1, label="Capital inicial")

    ax.set_xlabel("Día")
    ax.set_ylabel("Valor")
    ax.set_title("Evolución del portafolio")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if guardar:
        fig.savefig(guardar, dpi=150)

    plt.savefig("utils/plots/trayectoria.png")
    plt.show()
    return dias, valores, efectivos

def graficar_acciones(precios, nombres_activos=None):
  """
    Grafica la evolución del precio de m activos a lo largo de n días.

    Parámetros:
    precios (list o np.ndarray): Matriz de dimensiones (n, m) donde n son los días
                                 y m es la cantidad de activos.
    nombres_activos (list, opcional): Lista de longitud m con los nombres o tickers
                                      de cada activo.
    """
  # Convertir a arreglo de NumPy para facilitar el manejo de dimensiones
  datos = np.asarray(precios)

  if datos.ndim != 2:
    raise ValueError(
        "La matriz 'precios' debe ser bidimensional (n días x m activos)."
    )

  n_dias, m_activos = datos.shape

  # Generar etiquetas por defecto si no se proporcionan
  if nombres_activos is None:
    nombres_activos = [f"Activo {i + 1}" for i in range(m_activos)]
  elif len(nombres_activos) != m_activos:
    raise ValueError(
        f"Se esperaban {m_activos} nombres de activos, pero se recibieron"
        f" {len(nombres_activos)}."
    )

  # Crear el eje x (días)
  dias = np.arange(1, n_dias + 1)

  # Configurar y dibujar la gráfica
  plt.figure(figsize=(18, 6))

  for j in range(m_activos):
    plt.plot(
        dias, datos[:, j], linewidth=2, label=nombres_activos[j]
    )

  plt.title("Evolución del Valor de los Activos", fontsize=14, fontweight="bold")
  plt.xlabel("Días", fontsize=12)
  plt.ylabel("Precio / Valor", fontsize=12)
  plt.legend(title="Activos", loc="best")
  plt.grid(True, linestyle="--", alpha=0.6)
  plt.tight_layout()
  plt.savefig("utils/plots/acciones.png")
  #plt.show()