import numpy as np
from math import floor, ceil


class ProblemaTrading:

    def __init__(
        self,
        precios,
        efectivo_inicial=10_000,
        cantidades_iniciales=None,
        ventana=20,
        porcentaje = 0.15
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
        Genera todas las operaciones factibles para el día actual.

        Formato:
        (operacion, indice_activo, cantidad)
        """

        t, efectivo, cantidades = estado

        if self.test_objetivo(estado):
            return []

        acciones = [("MANTENER", None, 0)]

        for i in range(self.n_activos):

            precio = self.precios[t, i]

            # Compras: cualquier cantidad entera asequible.
            max_compra = floor(efectivo / precio)

            for n in range(1, max_compra + 1):
                acciones.append(("COMPRAR", i, n))

            # Ventas: cualquier cantidad entera poseída.
            for n in range(1, cantidades[i] + 1):
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

    def costo(self, estado, accion):
        """
        Convierte la maximización del valor del portafolio
        en una minimización del costo acumulado.

        Costo = valor actual - valor del sucesor.
        """

        sucesor = self.transicion(estado, accion)

        return (
            self.valor_portafolio(estado)
            - self.valor_portafolio(sucesor)
        )

    # --------------------------------------------------
    # Heurística
    # --------------------------------------------------

    def heuristica(self, estado):
        """
        Estima el costo restante hasta el final del horizonte.

        Utiliza medias móviles para proyectar los precios.
        """

        t, efectivo, cantidades = estado

        valor_actual = self.valor_portafolio(estado)

        if self.test_objetivo(estado):
            return 0.0

        k = self.ventana

        if t < 2 * k:
            k = min(k, (t+1)//2)
            if k < 2:
                return 0.0

        dias_restantes = self.n_dias - 1 - t

        if dias_restantes <= 0:
            return 0.0

        precios_historicos = self.precios[:t + 1]

        valor_estimado = efectivo

        for i in range(self.n_activos):

            ma_reciente = np.mean(
                precios_historicos[-k:, i]
            )

            ma_anterior = np.mean(
                precios_historicos[-2*k:-k, i]
            )

            rendimiento = (
                (ma_reciente / ma_anterior) ** (1 / k) - 1
            )

            precio_estimado = (
                precios_historicos[-1, i]
                * (1 + rendimiento) ** dias_restantes
            )

            valor_estimado += cantidades[i] * precio_estimado

        # Costo restante estimado:
        # valor actual - valor terminal proyectado.
        return valor_actual - valor_estimado

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

            valor = self.heuristica(nuevo_estado)

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