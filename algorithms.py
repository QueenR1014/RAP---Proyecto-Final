from types import MethodType
from math import inf
from utils.queue import ListaPrioritaria

def A_star(problema, heuristica, W=1):

    # Asignar la heurística al ambiente.
    #problema.heuristica = MethodType(heuristica, problema)

    # --------------------------------------------------
    # Inicialización
    # --------------------------------------------------

    nodo = problema.nodo_inicial()

    s = nodo.estado
    cod = problema.codigo(s)

    frontera = ListaPrioritaria()
    frontera.push(nodo, W * problema.heuristica(s))

    # Mejor costo acumulado conocido para cada estado.
    explorados = {cod: 0}

    # Información para reconstruir el camino.
    padres = {}

    # --------------------------------------------------
    # Búsqueda
    # --------------------------------------------------

    while not frontera.is_empty():
        nodo, prioridad = frontera.pop(with_value=True)
        print(
            f"Día explorado: {nodo.estado[0]} | "
            f"Frontera: {len(frontera.pila)}"
        )
        s = nodo.estado
        cod = problema.codigo(s)

        g = explorados[cod]
        h = problema.heuristica(s)

        # Ignorar entradas desactualizadas de la frontera.
        if prioridad > g + W * h + 1e-9:
            continue

        # Verificar objetivo.
        if problema.test_objetivo(s):
            acciones = reconstruir_camino(cod, padres)
            return nodo, acciones

        # Generar sucesores.
        for accion, hijo in nodo.generar_hijos():

            estado_hijo = hijo.estado
            cod_hijo = problema.codigo(estado_hijo)

            # Costo acumulado.
            nuevo_costo = (
                g + problema.costo(s, accion)
            )

            # Evaluar si encontramos un camino mejor.
            if nuevo_costo < explorados.get(cod_hijo, inf):

                explorados[cod_hijo] = nuevo_costo

                padres[cod_hijo] = (cod, accion)

                # Evaluación f(n) = g(n) + W*h(n).
                f = (
                    nuevo_costo
                    + W * problema.heuristica(estado_hijo)
                )

                frontera.push(hijo, f)

    return None


def reconstruir_camino(codigo, padres):

    acciones = []

    while codigo in padres:

        codigo_padre, accion = padres[codigo]

        acciones.append(accion)

        codigo = codigo_padre

    return acciones[::-1]
