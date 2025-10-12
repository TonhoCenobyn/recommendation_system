class Preferencias:
    def __init__(self, alpha=0.50):
        self.alpha = alpha
        self.user_genero = None
        self.user_diretor = None
        self.user_pais = None

    def adicionar(self, genero, diretor, pais, vect_genero, vect_diretor, vect_pais):
        novo_genero = vect_genero.transform([genero])
        novo_diretor = vect_diretor.transform([diretor])
        novo_pais = vect_pais.transform([pais])

        if self.user_genero is None:
            self.user_genero = novo_genero
            self.user_diretor = novo_diretor
            self.user_pais = novo_pais
        else:
            self.user_genero = self.alpha * novo_genero + (1 - self.alpha) * self.user_genero
            self.user_diretor = self.alpha * novo_diretor + (1 - self.alpha) * self.user_diretor
            self.user_pais = self.alpha * novo_pais + (1 - self.alpha) * self.user_pais
