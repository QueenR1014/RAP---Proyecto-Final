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