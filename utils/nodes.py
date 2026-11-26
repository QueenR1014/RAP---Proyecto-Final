class NodoTrading:

    def __init__(self, estado, ambiente):

        self.estado = estado
        self.ambiente = ambiente

    def generar_hijos(self):

        return self.ambiente.generar_hijos(self.estado)

    def __repr__(self):

        t, efectivo, cantidades = self.estado

        return (
            f"NodoTrading("
            f"dia={t}, "
            f"efectivo={efectivo:.2f}, "
            f"cantidades={list(cantidades)}"
            f")"
        )

import math
import random
from math import inf


class NodoMCTS:
    def __init__(self, estado, padre=None, accion=None,
                 acciones_sin_probar=None, ganancia=0.0, terminal=False):
        self.estado = estado
        self.padre = padre
        self.accion = accion                    # acción que llevó a este nodo
        self.hijos = []
        self.acciones_sin_probar = acciones_sin_probar or []
        self.ganancia = ganancia                # retorno acumulado desde la raíz
        self.terminal = terminal
        self.visitas = 0
        self.recompensa = 0.0                   # suma de retornos simulados


def _recompensa_paso(problema, estado, accion):
    """
    Retorno de un paso: ganancia de valor menos penalización.
    Es el negativo de problema.costo(), pero devuelve también el sucesor
    para no llamar dos veces a transicion.
    """
    sucesor = problema.transicion(estado, accion)
    r = problema.valor_portafolio(sucesor) - problema.valor_portafolio(estado)

    if hasattr(problema, "penalizacion_concentracion"):
        r -= problema.penalizacion_concentracion(sucesor)

    return sucesor, r


def _simular(problema, estado, rng, horizonte):
    """
    Rollout con política aleatoria. Si se corta por horizonte antes del
    final, el resto se estima con la heurística (-h = ganancia futura esperada).
    """
    total = 0.0
    pasos = 0

    while not problema.test_objetivo(estado):
        if horizonte is not None and pasos >= horizonte:
            total -= problema.heuristica(estado)
            break

        accion = rng.choice(problema.acciones_aplicables(estado))
        estado, r = _recompensa_paso(problema, estado, accion)
        total += r
        pasos += 1

    return total


def _mejor_hijo_uct(nodo, c, lo, hi):
    rango = (hi - lo) if hi > lo else 1.0
    log_n = math.log(nodo.visitas)

    mejor, mejor_val = None, -inf
    for h in nodo.hijos:
        q = (h.recompensa / h.visitas - lo) / rango     # media normalizada [0,1]
        u = q + c * math.sqrt(log_n / h.visitas)
        if u > mejor_val:
            mejor, mejor_val = h, u

    return mejor


def mcts_buscar(problema, estado_inicial, iteraciones=2000, c=1.0,
                horizonte_rollout=30, semilla=None):
    """
    Corre MCTS desde estado_inicial y devuelve (mejor_accion, raiz).
    La mejor acción es la del hijo más visitado.
    """
    rng = random.Random(semilla)

    raiz = NodoMCTS(
        estado_inicial,
        acciones_sin_probar=list(problema.acciones_aplicables(estado_inicial)),
        terminal=problema.test_objetivo(estado_inicial),
    )
    if raiz.terminal:
        return None, raiz

    rng.shuffle(raiz.acciones_sin_probar)
    lo, hi = inf, -inf      # rango de retornos vistos, para normalizar UCT

    for _ in range(iteraciones):

        # 1) Selección
        nodo = raiz
        while not nodo.terminal and not nodo.acciones_sin_probar and nodo.hijos:
            nodo = _mejor_hijo_uct(nodo, c, lo, hi) if nodo.visitas > 0 \
                else rng.choice(nodo.hijos)

        # 2) Expansión
        if not nodo.terminal and nodo.acciones_sin_probar:
            accion = nodo.acciones_sin_probar.pop()
            estado_hijo, r = _recompensa_paso(problema, nodo.estado, accion)
            terminal = problema.test_objetivo(estado_hijo)

            acciones = [] if terminal else list(problema.acciones_aplicables(estado_hijo))
            rng.shuffle(acciones)

            hijo = NodoMCTS(estado_hijo, padre=nodo, accion=accion,
                            acciones_sin_probar=acciones,
                            ganancia=nodo.ganancia + r, terminal=terminal)
            nodo.hijos.append(hijo)
            nodo = hijo

        # 3) Simulación
        retorno = nodo.ganancia + _simular(problema, nodo.estado, rng, horizonte_rollout)

        lo, hi = min(lo, retorno), max(hi, retorno)

        # 4) Retropropagación
        while nodo is not None:
            nodo.visitas += 1
            nodo.recompensa += retorno
            nodo = nodo.padre

    mejor = max(raiz.hijos, key=lambda h: h.visitas)
    return mejor.accion, raiz


def ejecutar_mcts(problema, estado_inicial=None, iteraciones=2000, c=1.0,
                  horizonte_rollout=30, semilla=0, verbose_cada=20):
    """
    Planifica un episodio completo: en cada día corre MCTS, ejecuta la
    mejor acción y avanza. Devuelve (estado_final, acciones), con el mismo
    formato de acciones que A*, así que sirve con graficar_trayectoria.
    """
    estado = estado_inicial if estado_inicial is not None else problema.estado_inicial
    acciones = []

    while not problema.test_objetivo(estado):
        accion, _ = mcts_buscar(problema, estado, iteraciones, c,
                                horizonte_rollout,
                                semilla=None if semilla is None else semilla + estado[0])
        acciones.append(accion)
        estado = problema.transicion(estado, accion)

        if verbose_cada and estado[0] % verbose_cada == 0:
            print(f"Día {estado[0]} | valor: {problema.valor_portafolio(estado):.2f}")

    return estado, acciones
    