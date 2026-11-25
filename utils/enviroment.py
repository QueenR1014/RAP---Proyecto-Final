import numpy as np
from math import floor, ceil


class ProblemaTrading:

    def __init__(
        self,
        precios,
        efectivo_inicial=10_000,
        cantidades_iniciales=None,
        ventana=20,
        porcentaje = 0.15,
        lambda_conc=0.002, 
        w_max=0.3
    ):
        """
        Ambiente para el problema de trading.

        precios: matriz (dias, activos).
        efectivo_inicial: capital disponible al inicio.
        cantidades_iniciales: acciones iniciales de cada activo.
        ventana: ventana para estimar la tendencia de precios.
        """

        self.precios = np.asarray(precios, dtype=float)

        if self.precios.ndim != 2:
            raise ValueError("La matriz de precios debe ser bidimensional.")

        if not np.all(np.isfinite(self.precios)):
            raise ValueError("La matriz contiene precios no válidos.")

        if np.any(self.precios <= 0):
            raise ValueError("Los precios deben ser positivos.")

        self.n_dias, self.n_activos = self.precios.shape
        self.ventana = ventana

        if cantidades_iniciales is None:
            cantidades_iniciales = np.zeros(
                self.n_activos, dtype=int
            )

        cantidades_iniciales = np.asarray(
            cantidades_iniciales, dtype=int
        )

        if cantidades_iniciales.shape != (self.n_activos,):
            raise ValueError("Dimensiones iniciales incorrectas.")

        self.estado_inicial = (
            0,
            float(efectivo_inicial),
            tuple(cantidades_iniciales)
        )

        self.porcentaje = porcentaje

        self.lambda_conc = lambda_conc   # intensidad de la penalización
        self.w_max = w_max               # peso máximo tolerado por activo
    # --------------------------------------------------
    # Valoración del portafolio
    # --------------------------------------------------

    def valor_portafolio(self, estado):
        """
        Valor del portafolio a precios del día del estado.
        En el estado terminal utiliza el último precio disponible.
        """

        t, efectivo, cantidades = estado

        indice = min(t, self.n_dias - 1)

        return (
            efectivo
            + np.dot(cantidades, self.precios[indice])
        )

    # --------------------------------------------------
    # Acciones aplicables
    # --------------------------------------------------

    def acciones_aplicables(self, estado):
        """
        Genera las operaciones factibles para el día actual.
        Las cantidades se discretizan en fracciones del máximo posible
        para mantener acotado el factor de ramificación.

        Formato:
        (operacion, indice_activo, cantidad)
        """

        t, efectivo, cantidades = estado

        if self.test_objetivo(estado):
            return []

        acciones = [("MANTENER", None, 0)]

        fracciones = (0.25, 0.5, 0.75, 1.0)

        for i in range(self.n_activos):

            precio = self.precios[t, i]

            # Compras: fracciones del máximo asequible.
            max_compra = floor((efectivo + 1e-9) / precio)
            compras = {floor(max_compra * f) for f in fracciones}

            for n in sorted(compras):
                if n > 0:
                    acciones.append(("COMPRAR", i, n))

            # Ventas: fracciones de lo poseído.
            max_venta = int(cantidades[i])
            ventas = {floor(max_venta * f) for f in fracciones}

            for n in sorted(ventas):
                if n > 0:
                    acciones.append(("VENDER", i, n))

        return acciones

    # --------------------------------------------------
    # Función de transición
    # --------------------------------------------------

    def transicion(self, estado, accion):
        """
        Aplica una operación y avanza exactamente un día.
        """

        t, efectivo, cantidades = estado

        if self.test_objetivo(estado):
            raise ValueError("El estado ya es terminal.")

        operacion, activo, n = accion

        nuevo_efectivo = efectivo
        nuevas_cantidades = list(cantidades)

        if operacion == "COMPRAR":

            precio = self.precios[t, activo]
            costo = n * precio

            if n <= 0 or costo > efectivo + 1e-9:
                raise ValueError("Compra no válida.")

            nuevo_efectivo -= costo
            nuevas_cantidades[activo] += n

        elif operacion == "VENDER":

            if n <= 0 or n > cantidades[activo]:
                raise ValueError("Venta no válida.")

            precio = self.precios[t, activo]

            nuevo_efectivo += n * precio
            nuevas_cantidades[activo] -= n

        elif operacion != "MANTENER":
            raise ValueError("Operación desconocida.")

        return (
            t + 1,
            nuevo_efectivo,
            tuple(nuevas_cantidades)
        )

    # --------------------------------------------------
    # Prueba de objetivo
    # --------------------------------------------------

    def test_objetivo(self, estado):
        """
        El objetivo se alcanza al terminar los días disponibles.
        """

        t, _, _ = estado

        return t >= self.n_dias

    # --------------------------------------------------
    # Costo de transición
    # --------------------------------------------------

    def penalizacion_concentracion(self, estado):
        """
        Costo diario por concentración: lambda * V * sum(max(0, w_i - w_max)^2).
        """
        t, efectivo, cantidades = estado
        idx = min(t, self.n_dias - 1)

        posiciones = np.asarray(cantidades, dtype=float) * self.precios[idx]
        valor = efectivo + posiciones.sum()

        if valor <= 0:
            return 0.0

        pesos = posiciones / valor
        exceso = np.maximum(0.0, pesos - self.w_max)

        return self.lambda_conc * valor * np.sum(exceso ** 2)

    def costo(self, estado, accion):
        sucesor = self.transicion(estado, accion)

        return (
            self.valor_portafolio(estado)
            - self.valor_portafolio(sucesor)
            + self.penalizacion_concentracion(sucesor)
        )
    
    # --------------------------------------------------
    # Heurística
    # --------------------------------------------------

    def _factores_crecimiento(self, t):
        """
        Factores de crecimiento proyectados hasta el último día para cada
        activo, usando solo precios hasta t (sin mirar el futuro).
        Devuelve (factores_por_activo, factor_efectivo).
        """
        if not hasattr(self, "_cache_fact"):
            self._cache_fact = {}

        if t in self._cache_fact:
            return self._cache_fact[t]

        dias_restantes = self.n_dias - 1 - t
        k = min(self.ventana, (t + 1) // 2)

        if k < 1 or dias_restantes <= 0:
            tasas = np.zeros(self.n_activos)
        else:
            hist = self.precios[:t + 1]
            ma_rec = hist[-k:].mean(axis=0)
            ma_ant = hist[-2 * k:-k].mean(axis=0)
            tasas = (ma_rec / ma_ant) ** (1.0 / k) - 1.0
            tasas = np.clip(tasas, -0.05, 0.05)   # evita proyecciones absurdas

        factores = (1.0 + tasas) ** max(dias_restantes, 0)
        factor_efectivo = max(1.0, factores.max())  # el efectivo puede invertirse

        self._cache_fact[t] = (factores, factor_efectivo)
        return self._cache_fact[t]


    def heuristica(self, estado):
        if self.test_objetivo(estado):
            return 0.0

        t, efectivo, cantidades = estado
        factores, f_efectivo = self._factores_crecimiento(t)

        valor_actual = self.valor_portafolio(estado)
        valor_estimado = (
            efectivo * f_efectivo
            + np.dot(cantidades, self.precios[t] * factores)
        )

        dias_restantes = self.n_dias - 1 - t
        pen_restante = self.penalizacion_concentracion(estado) * dias_restantes

        return valor_actual - valor_estimado + pen_restante

    # --------------------------------------------------
    # Identificación de estados
    # --------------------------------------------------

    def codigo(self, estado):
        """
        Representación hashable para detectar estados repetidos.
        """

        t, efectivo, cantidades = estado

        return (
            t,
            round(efectivo, 8),
            tuple(cantidades)
        )

    
    # --------------------------------------------------
    #  Generación de estados
    # --------------------------------------------------

    def generar_hijos(self, estado):
        """
        Genera los nodos sucesores de un estado.
        """

        from utils.nodes import NodoTrading

        hijos = []

        acciones = self.acciones_aplicables(estado)

        evaluados = []

        for accion in acciones:

            nuevo_estado = self.transicion(estado, accion)

            valor = self.costo(estado, accion) + self.heuristica(nuevo_estado)   # f del hijo

            evaluados.append((accion, nuevo_estado, valor))

        # Ordenar por valor heurístico descendente.
        evaluados.sort(
            key=lambda x: x[2]
        )

        # Conservar el porcentaje más prometedor.
        n_conservar = max(
            1,
            ceil(len(evaluados) * self.porcentaje)
        )
        if estado[0] >= 23:
            print(
                f"Día {estado[0]} | "
                f"Acciones: {len(acciones)} | "
                f"Hijos: {n_conservar}"
            )

        for accion, nuevo_estado, valor in evaluados[:n_conservar]:

            hijo = NodoTrading(
                nuevo_estado,
                self
            )

            hijos.append((accion, hijo))

        return hijos


    def nodo_inicial(self):
        """
        Construye el nodo correspondiente al estado inicial.
        """

        from utils.nodes import NodoTrading

        return NodoTrading(
            self.estado_inicial,
            self
        )