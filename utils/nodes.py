import numpy as np
from math import floor, ceil


class NodoTrading:

    MANTENER = ("MANTENER", None, 0)

    def __init__(
        self,
        precios,
        t=0,
        efectivo=10_000,
        cantidades=None,
        ventana=20,
        porcentaje=0.15,
        calcular_acciones=True
    ):
        """
        precios: matriz de dimensiones (dias, activos).
        t: día actual del nodo.
        efectivo: dinero disponible.
        cantidades: acciones poseídas de cada activo.
        ventana: ventana utilizada para calcular la tendencia.
        porcentaje: proporción de acciones candidatas que se conservan.
        """

        self.precios = np.asarray(precios, dtype=float)

        if self.precios.ndim != 2:
            raise ValueError("La matriz de precios debe ser bidimensional.")

        if not np.all(np.isfinite(self.precios)):
            raise ValueError("La matriz contiene precios no válidos.")

        if np.any(self.precios <= 0):
            raise ValueError("Todos los precios deben ser positivos.")

        self.n_dias, self.n_activos = self.precios.shape

        if not 0 <= t <= self.n_dias:
            raise ValueError("El día está fuera del horizonte de simulación.")

        self.t = t
        self.efectivo = float(efectivo)

        self.cantidades = (
            np.zeros(self.n_activos, dtype=int)
            if cantidades is None
            else np.asarray(cantidades, dtype=int).copy()
        )

        if self.cantidades.shape != (self.n_activos,):
            raise ValueError("La cantidad de activos no coincide con los precios.")

        if np.any(self.cantidades < 0):
            raise ValueError("No se permiten cantidades negativas.")

        self.ventana = ventana
        self.porcentaje = porcentaje

        self.acciones_validas = []

        if calcular_acciones and not self.es_terminal():
            self.acciones_validas = self.generar_acciones_validas()

    # --------------------------------------------------
    # Estado
    # --------------------------------------------------

    def es_terminal(self):
        return self.t >= self.n_dias

    def valor_portafolio(self, hasta=None):
        """
        Calcula el valor del portafolio usando los precios
        observados hasta el día indicado.
        """

        if hasta is None:
            hasta = min(self.t, self.n_dias - 1)

        return (
            self.efectivo
            + np.dot(self.cantidades, self.precios[hasta])
        )

    # --------------------------------------------------
    # Heurística
    # --------------------------------------------------

    def heuristica(self, hasta=None):
        """
        Estima el valor terminal del portafolio utilizando
        la tendencia reciente de los precios.

        Solo utiliza información hasta el día 'hasta'.
        """

        if hasta is None:
            hasta = min(self.t, self.n_dias - 1)

        valor_actual = self.valor_portafolio(hasta)

        k = self.ventana

        # Necesitamos dos ventanas completas para estimar
        # una tendencia.
        if hasta + 1 < 2 * k:
            return valor_actual

        dias_restantes = self.n_dias - 1 - hasta

        if dias_restantes <= 0:
            return valor_actual

        precios_historicos = self.precios[:hasta + 1]

        precios_estimados = np.zeros(self.n_activos)

        for i in range(self.n_activos):

            # Promedio de los últimos k días.
            ma_reciente = np.mean(
                precios_historicos[-k:, i]
            )

            # Promedio de los k días anteriores.
            ma_anterior = np.mean(
                precios_historicos[-2*k:-k, i]
            )

            # Crecimiento diario estimado.
            rendimiento = (
                (ma_reciente / ma_anterior) ** (1 / k) - 1
            )

            precio_actual = precios_historicos[-1, i]

            # Proyección hasta el último día.
            precios_estimados[i] = (
                precio_actual
                * (1 + rendimiento) ** dias_restantes
            )

        return (
            self.efectivo
            + np.dot(self.cantidades, precios_estimados)
        )

    # --------------------------------------------------
    # Generación de acciones
    # --------------------------------------------------

    def generar_acciones_candidatas(self):
        """
        Genera todas las acciones posibles para el día actual.

        Formato:
        (operacion, indice_activo, cantidad)
        """

        if self.es_terminal():
            return []

        acciones = [self.MANTENER]

        for i in range(self.n_activos):

            precio = self.precios[self.t, i]

            # ------------------------------------------
            # Comprar
            # ------------------------------------------

            max_compra = floor(self.efectivo / precio)

            for cantidad in range(1, max_compra + 1):

                acciones.append(
                    ("COMPRAR", i, cantidad)
                )

            # ------------------------------------------
            # Vender
            # ------------------------------------------

            max_venta = self.cantidades[i]

            for cantidad in range(1, max_venta + 1):

                acciones.append(
                    ("VENDER", i, cantidad)
                )

        return acciones

    # --------------------------------------------------
    # Transición entre estados
    # --------------------------------------------------

    def aplicar_accion(self, accion, calcular_acciones=True):
        """
        Aplica una acción y devuelve un nuevo nodo
        correspondiente al día siguiente.
        """

        if self.es_terminal():
            raise ValueError("No se pueden aplicar acciones a un nodo terminal.")

        operacion, activo, cantidad = accion

        nuevo_efectivo = self.efectivo
        nuevas_cantidades = self.cantidades.copy()

        if operacion == "COMPRAR":

            precio = self.precios[self.t, activo]
            costo = cantidad * precio

            if cantidad <= 0 or costo > self.efectivo + 1e-9:
                raise ValueError("Compra no válida.")

            nuevo_efectivo -= costo
            nuevas_cantidades[activo] += cantidad

        elif operacion == "VENDER":

            precio = self.precios[self.t, activo]

            if cantidad <= 0 or cantidad > self.cantidades[activo]:
                raise ValueError("Venta no válida.")

            nuevo_efectivo += cantidad * precio
            nuevas_cantidades[activo] -= cantidad

        elif operacion != "MANTENER":
            raise ValueError("Operación desconocida.")

        # El siguiente nodo corresponde al siguiente día.
        return NodoTrading(
            precios=self.precios,
            t=self.t + 1,
            efectivo=nuevo_efectivo,
            cantidades=nuevas_cantidades,
            ventana=self.ventana,
            porcentaje=self.porcentaje,
            calcular_acciones=calcular_acciones
        )

    # --------------------------------------------------
    # Selección de acciones mediante heurística
    # --------------------------------------------------

    def generar_acciones_validas(self):
        """
        Conserva el porcentaje de acciones con mejor
        evaluación heurística.
        """

        acciones = self.generar_acciones_candidatas()

        evaluadas = []

        for accion in acciones:

            sucesor = self.aplicar_accion(
                accion,
                calcular_acciones=False
            )

            # Evaluamos todos los sucesores utilizando
            # únicamente información disponible en el día actual.
            valor = sucesor.heuristica(hasta=self.t)

            evaluadas.append((accion, valor))

        # Orden descendente: mayor valor estimado primero.
        evaluadas.sort(
            key=lambda x: x[1],
            reverse=True
        )

        # Conservar el porcentaje más prometedor.
        n_conservar = max(
            1,
            ceil(len(evaluadas) * self.porcentaje)
        )

        return [
            accion
            for accion, valor in evaluadas[:n_conservar]
        ]

    # --------------------------------------------------
    # Generación de sucesores
    # --------------------------------------------------

    def generar_sucesores(self):

        sucesores = []

        for accion in self.acciones_validas:

            nuevo_nodo = self.aplicar_accion(
                accion,
                calcular_acciones=True
            )

            sucesores.append(nuevo_nodo)

        return sucesores

    # --------------------------------------------------
    # Representación
    # --------------------------------------------------

    def __repr__(self):

        return (
            f"NodoTrading("
            f"dia={self.t}, "
            f"efectivo={self.efectivo:.2f}, "
            f"cantidades={self.cantidades.tolist()}, "
            f"valor={self.valor_portafolio():.2f}"
            f")"
        )