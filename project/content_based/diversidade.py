from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# MICRODIVERSIDADE
def diversidade_aspecto(matriz, recomendados):
    recomendados = list(set(recomendados))
    features = matriz[recomendados]
    k = features.shape[0]

    if k < 2:
        return np.nan

    sim_matrix = cosine_similarity(features)
    i, j = np.triu_indices(k, k=1)
    media_sim = sim_matrix[i, j].mean()
    return float(1 - media_sim)

def diversidade_rodada(ids, m_gen, m_dir, m_pais, pesos=(1/3, 1/3, 1/3)):
    div_genero = diversidade_aspecto(m_gen, ids)
    div_diretor = diversidade_aspecto(m_dir, ids)
    div_pais = diversidade_aspecto(m_pais, ids)
    div_total = (pesos[0] * div_genero +
                 pesos[1] * div_diretor +
                 pesos[2] * div_pais)
    return {
        "div_genero": div_genero,
        "div_diretor": div_diretor,
        "div_pais": div_pais,
        "div_total": div_total
    }

# MACRODIVERSIDADE
def macrodiversidade_aspecto(lista_recomendacoes, matriz):
    # Combina todas as recomendações de todas as rodadas
    todos_ids = []
    for rodada in lista_recomendacoes:
        todos_ids.extend(rodada)

    # Remove duplicados
    todos_ids = list(set(todos_ids))

    # Caso tenha menos de 2 elementos, não dá para calcular similaridade
    if len(todos_ids) < 2:
        return np.nan

    features = matriz[todos_ids]
    sim_matrix = cosine_similarity(features)
    i, j = np.triu_indices(len(todos_ids), k=1)
    media_sim = sim_matrix[i, j].mean()
    return float(1 - media_sim)


def macrodiversidade_rodadas(lista_recomendacoes, m_gen, m_dir, m_pais, pesos=(1/3, 1/3, 1/3), ultimas_x=None):
    if ultimas_x is not None:
        lista_recomendacoes = lista_recomendacoes[-ultimas_x:]

    div_genero = macrodiversidade_aspecto(lista_recomendacoes, m_gen)
    div_diretor = macrodiversidade_aspecto(lista_recomendacoes, m_dir)
    div_pais = macrodiversidade_aspecto(lista_recomendacoes, m_pais)

    div_total = (pesos[0] * div_genero +
                 pesos[1] * div_diretor +
                 pesos[2] * div_pais)

    return {
        "macro_div_genero": div_genero,
        "macro_div_diretor": div_diretor,
        "macro_div_pais": div_pais,
        "macro_div_total": div_total
    }