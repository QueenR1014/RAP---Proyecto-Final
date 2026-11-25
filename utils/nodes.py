class NodoTrading:
    """
    Nodo de búsqueda.

    El nodo conserva únicamente su estado y una referencia al ambiente
    responsable de la lógica del problema. La generación de hijos se delega
    por completo al ambiente.
    """

    def __init__(self, estado, ambiente):
        if len(estado) != 3:
            raise ValueError(
                "El estado debe tener la forma (dia, efectivo, cantidades)."
            )

        dia, efectivo, cantidades = estado
        self.estado = (int(dia), float(efectivo), tuple(cantidades))
        self.ambiente = ambiente

    def generar_hijos(self):
        """Delega al ambiente la generación de los nodos sucesores."""
        return self.ambiente.generar_hijos(self.estado)

    def __repr__(self):
        dia, efectivo, cantidades = self.estado
        return (
            f"NodoTrading(dia={dia}, efectivo={efectivo:.2f}, "
            f"cantidades={list(cantidades)})"
        )
