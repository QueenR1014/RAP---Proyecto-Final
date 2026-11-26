import numpy as np
import matplotlib.pyplot as plt


def graficar_trayectoria(problema, acciones, nombres_activos=None, guardar=None):
    """
    Reconstruye la trayectoria a partir de las acciones (A* o MCTS) y grafica:
      1) valor del portafolio, efectivo y valor invertido,
      2) composición del portafolio en dinero (efectivo + cada activo),
      3) cantidad de acciones de cada activo.

    nombres_activos: lista de nombres; si es None se usan los de las columnas
    del DataFrame original (si existen) o "Activo i".
    """

    if nombres_activos is None:
        nombres_activos = [f"Activo {i}" for i in range(problema.n_activos)]

    estado = problema.estado_inicial

    dias = []
    valores = []
    efectivos = []
    cantidades_dia = []     # acciones poseídas al cierre de cada día
    posiciones_dia = []     # valor en dinero de cada activo al cierre

    for accion in acciones:
        t, _, _ = estado

        siguiente = problema.transicion(estado, accion)
        _, efectivo_sig, cantidades_sig = siguiente

        cant = np.asarray(cantidades_sig, dtype=float)
        pos = cant * problema.precios[t]            # dinero en cada activo
        valor = efectivo_sig + pos.sum()

        dias.append(t)
        valores.append(valor)
        efectivos.append(efectivo_sig)
        cantidades_dia.append(cant)
        posiciones_dia.append(pos)

        estado = siguiente

    dias = np.array(dias)
    valores = np.array(valores)
    efectivos = np.array(efectivos)
    cantidades_dia = np.array(cantidades_dia)       # (dias, activos)
    posiciones_dia = np.array(posiciones_dia)       # (dias, activos)

    fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # --- Panel 1: valor, efectivo e invertido ---
    ax = axs[0]
    ax.plot(dias, valores, label="Valor del portafolio", linewidth=2)
    ax.plot(dias, efectivos, label="Efectivo", linewidth=2)
    ax.plot(dias, valores - efectivos, label="Valor invertido",
            linewidth=1.5, linestyle="--")
    ax.axhline(problema.estado_inicial[1], color="gray",
               linestyle=":", linewidth=1, label="Capital inicial")
    ax.set_ylabel("Valor")
    ax.set_title("Evolución del portafolio")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    # --- Panel 2: composición en dinero (áreas apiladas) ---
    ax = axs[1]
    capas = [efectivos] + [posiciones_dia[:, i] for i in range(problema.n_activos)]
    etiquetas = ["Efectivo"] + list(nombres_activos)
    ax.stackplot(dias, *capas, labels=etiquetas, alpha=0.85)
    ax.set_ylabel("Valor por componente")
    ax.set_title("Composición del portafolio (dinero)")
    ax.legend(loc="upper left", ncol=3, fontsize=8)
    ax.grid(alpha=0.3)

    # --- Panel 3: cantidad de acciones por activo ---
    ax = axs[2]
    for i in range(problema.n_activos):
        # Solo se dibujan los activos que se llegaron a poseer.
        if cantidades_dia[:, i].max() > 0:
            ax.step(dias, cantidades_dia[:, i], where="post",
                    label=nombres_activos[i], linewidth=1.5)
    ax.set_xlabel("Día")
    ax.set_ylabel("Acciones poseídas")
    ax.set_title("Cantidad de acciones por activo")
    ax.legend(loc="upper left", ncol=3, fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()

    if guardar:
        fig.savefig(guardar, dpi=150)

    plt.show()

    return dias, valores, efectivos, cantidades_dia, posiciones_dia

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