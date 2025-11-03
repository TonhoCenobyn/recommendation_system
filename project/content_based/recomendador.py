from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

def normalizar(sim):
    scaler = MinMaxScaler()
    return scaler.fit_transform(sim.T).T

def gerar_recomendacao(preferencias, m_gen, m_dir, m_pais):
    """
    Retorna:
      - sim_scores: lista de tuples (idx, score) ordenada desc
      - sim_genero, sim_diretor, sim_pais: matrizes de similaridade normalizadas (1 x N)
    """
    sim_genero = normalizar(cosine_similarity(preferencias.user_genero, m_gen))
    sim_diretor = normalizar(cosine_similarity(preferencias.user_diretor, m_dir))
    sim_pais = normalizar(cosine_similarity(preferencias.user_pais, m_pais))

    user_similaridade = (sim_genero + sim_diretor + sim_pais) / 3.0
    sim_scores = list(enumerate(user_similaridade[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    return sim_scores, sim_genero, sim_diretor, sim_pais


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


