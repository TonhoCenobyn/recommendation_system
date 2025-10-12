from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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

def diversidade_total(ids, m_gen, m_dir, m_pais, pesos=(1/3, 1/3, 1/3)):
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
