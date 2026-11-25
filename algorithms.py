from math import inf
from utils.queue import ListaPrioritaria


def A_star(problema, heuristica, W=1, verbose_cada=1000):

    nodo = problema.nodo_inicial()
    s = nodo.estado
    cod = problema.codigo(s)

    frontera = ListaPrioritaria()
    frontera.push(nodo, W * heuristica(s))

    mejor_g = {cod: 0.0}      # mejor costo acumulado conocido
    padres = {}               # cod_hijo -> (cod_padre, accion)
    cerrados = set()

    expansiones = 0

    while not frontera.is_empty():

        nodo, prioridad = frontera.pop(with_value=True)
        s = nodo.estado
        cod = problema.codigo(s)

        # Ya expandido con un costo igual o mejor: entrada obsoleta.
        if cod in cerrados:
            continue

        g = mejor_g[cod]

        # Entrada obsoleta: se encontró un camino mejor después de insertarla.
        if prioridad > g + W * heuristica(s) + 1e-9:
            continue

        if problema.test_objetivo(s):
            return nodo, reconstruir_camino(cod, padres)

        cerrados.add(cod)
        expansiones += 1

        if verbose_cada and expansiones % verbose_cada == 0:
            print(f"Exp: {expansiones} | día: {s[0]} | frontera: {len(frontera.pila)}")

        for accion, hijo in nodo.generar_hijos():

            e_hijo = hijo.estado
            cod_hijo = problema.codigo(e_hijo)

            nuevo_g = g + problema.costo(s, accion)

            if nuevo_g < mejor_g.get(cod_hijo, inf) - 1e-9:
                mejor_g[cod_hijo] = nuevo_g
                padres[cod_hijo] = (cod, accion)
                cerrados.discard(cod_hijo)   # reabrir si mejoró
                frontera.push(hijo, nuevo_g + W * heuristica(e_hijo))

    return None


def reconstruir_camino(codigo, padres):
    acciones = []
    while codigo in padres:
        codigo, accion = padres[codigo][0], padres[codigo][1]
        acciones.append(accion)
    return acciones[::-1]