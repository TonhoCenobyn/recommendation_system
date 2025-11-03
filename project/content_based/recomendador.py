from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

def normalizar(sim):
    scaler = MinMaxScaler()
    return scaler.fit_transform(sim.T).T

def gerar_recomendacao(preferencias, m_gen, m_dir, m_pais, nivel_aleatoriedade):
    """
    Retorna:
      - sim_scores: lista de tuples (idx, score) ordenada desc
      - sim_genero, sim_diretor, sim_pais: matrizes de similaridade normalizadas (1 x N)
    """
    sim_genero = cosine_similarity(preferencias.user_genero, m_gen)
    sim_diretor = cosine_similarity(preferencias.user_diretor, m_dir)
    sim_pais = cosine_similarity(preferencias.user_pais, m_pais)

    user_similaridade = (sim_genero + sim_diretor + sim_pais) / 3.0
    sim_scores = list(enumerate(user_similaridade[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    return sim_scores, sim_genero, sim_diretor, sim_pais


import random  # <-- ADICIONE ESTE IMPORT
from sklearn.metrics.pairwise import cosine_similarity


# Se você tiver a função normalizar, ela deve estar aqui
# from sklearn.preprocessing import MinMaxScaler
# def normalizar(sim):
#     scaler = MinMaxScaler()
#     return scaler.fit_transform(sim.T).T

def gerar_recomendacao_controlavel(preferencias, m_gen, m_dir, m_pais, nivel_aleatoriedade, top_n):
    """
    Retorna:
      - sim_scores: lista de tuples (idx, score) ordenada desc
      - sim_genero, sim_diretor, sim_pais: matrizes de similaridade (1 x N)
    """

    sim_genero = cosine_similarity(preferencias.user_genero, m_gen)
    sim_diretor = cosine_similarity(preferencias.user_diretor, m_dir)
    sim_pais = cosine_similarity(preferencias.user_pais, m_pais)

    user_similaridade = (sim_genero + sim_diretor + sim_pais) / 3.0

    sim_scores = list(enumerate(user_similaridade[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    porcentagens_troca = {
        1: 0.20,  # 20%
        2: 0.50,  # 50%
        3: 0.80  # 80%
    }

    porcentagem = porcentagens_troca.get(nivel_aleatoriedade, 0.0)

    if porcentagem == 0.0:
        return sim_scores, sim_genero, sim_diretor, sim_pais

    quantidade_troca = int(top_n * porcentagem)

    total_items = len(sim_scores)
    if top_n + quantidade_troca > total_items:
        print("QUANTIDADE DE TROCA MAIOR QUE QUANTIDADE DE FILMES NA BASE")
        num_to_swap = total_items - top_n
        if num_to_swap <= 0:
            return sim_scores, sim_genero, sim_diretor, sim_pais

    modified_scores = sim_scores.copy()

    indices_trocados = random.sample(range(top_n), quantidade_troca)

    indices_novos = random.sample(range(top_n, total_items), quantidade_troca)

    for i in range(quantidade_troca):
        idx_top = indices_trocados[i]  # Posição de um item no topo
        idx_bottom = indices_novos[i]  # Posição de um item no fundo

        modified_scores[idx_top], modified_scores[idx_bottom] = modified_scores[idx_bottom], modified_scores[idx_top]

    return modified_scores, sim_genero, sim_diretor, sim_pais



def gerar_recomendacao_onehot(movies, genero, diretor, pais):
    """
    Similaridade baseada em match exato com One-Hot Encoding.
    """
    encoder = OneHotEncoder(sparse_output=True)
    matriz_total = encoder.fit_transform(movies[['genero','diretor','pais_origem']])
    user_total = encoder.transform(pd.DataFrame(
        [[genero, diretor, pais]],
        columns=['genero', 'diretor', 'pais_origem']
    ))
    # Similaridade coseno
    sim_total = cosine_similarity(user_total, matriz_total)

    # Ordenar por similaridade
    sim_scores = list(enumerate(sim_total[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    return sim_scores, sim_total


