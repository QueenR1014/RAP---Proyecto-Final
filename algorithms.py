from utils.enviroment import ProblemaTrading
from utils.nodes import NodoTrading

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
    ventana= 7
)

# --------------------------------------------------
# Búsqueda A*
# --------------------------------------------------

# Ejercicio 7a
def A_star(problema, heuristica, W):

    problema.heuristica = MethodType(heuristica, problema)

    # AQUÍ SU CÓDIGO
    s = problema.estado_inicial
    v = problema.heuristica(s)
    cod = problema.codigo(s)
    nodo = Nodo(s,None,None,0,cod)
    frontera = ListaPrioritaria()
    frontera.push(nodo, v)
    explorados = {cod:v}
    while not frontera.is_empty():
        nodo = frontera.pop()
        if problema.test_objetivo(nodo.estado): return nodo
        for accion in problema.acciones_aplicables(nodo.estado):
            hijo = nodo_hijo(problema, nodo, accion)
            s = hijo.estado
            v = problema.heuristica(s)
            cod = hijo.codigo
            c = hijo.costo_camino
            h = W * v + c
            if cod not in explorados.keys() or h < explorados[cod]:
                frontera.push(hijo, h)
                explorados[cod] = h
    return None
    # HASTA AQUÍ SU CÓDIGO