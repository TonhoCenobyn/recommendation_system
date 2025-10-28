import random

def inserir_entradas(movies, historico_entradas, nivel_aleatoriedade):
    parametros_aleatorios = random.sample(['genero', 'diretor', 'pais_origem'], nivel_aleatoriedade)

    def escolher(campo):
        if campo in parametros_aleatorios or len(historico_entradas[campo]) == 0:
            return random.choice(movies[campo].unique())
        else:
            return random.choice(historico_entradas[campo])

    genero = escolher("genero")
    diretor = escolher("diretor")
    pais = escolher("pais_origem")

    print(f"Gênero escolhido: {genero}")
    print(f"Diretor escolhido: {diretor}")
    print(f"País escolhido: {pais}")
    return genero, diretor, pais