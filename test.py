from utils.enviroment import ProblemaTrading

from algorithms import A_star, reconstruir_camino

import pandas as pd
# --------------------------------------------------
# Lectura de Precios
# --------------------------------------------------
precios = pd.read_csv("./data/stock_prices.csv").iloc[:252,1:]

#print(precios.shape)
#precios.info()

# --------------------------------------------------
# Creación del Ambiente
# --------------------------------------------------

problema = ProblemaTrading(
    precios=precios, 
    efectivo_inicial=10000,
    cantidades_iniciales=None,
    ventana= 20,
    porcentaje= 0.01
)


heuristica_trading = problema.heuristica
resultado = A_star(
    problema,
    heuristica_trading,
    W=1
)

if resultado is not None:

    nodo_final, acciones = resultado

    print("Estado final:", nodo_final.estado)
    print("Acciones:", acciones)

    print(
        "Valor final:",
        problema.valor_portafolio(nodo_final.estado)
    )

from utils.graphs import graficar_trayectoria, graficar_acciones

graficar_acciones(precios=precios.to_numpy(), nombres_activos=precios.columns.to_list())
graficar_trayectoria(problema=problema, acciones=resultado[1])