from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

#RELEVANCIA
def reajustar_relevancia(relevancias, referencia=0.7):
    if isinstance(relevancias, dict):
        return {
            chave: valor / referencia if referencia != 0 else valor
            for chave, valor in relevancias.items()
        }
    elif isinstance(relevancias, (int, float)):
        return relevancias / referencia if referencia != 0 else relevancias
    else:
        raise TypeError(f"Tipo não suportado em reajustar_relevancia: {type(relevancias)}")


# MICRODIVERSIDADE
def microdiversidade_parametro(matriz, recomendados):
    recomendados = list(recomendados)
    features = matriz[recomendados]
    k = features.shape[0]

    if k < 2:
        return np.nan

    sim_matrix = cosine_similarity(features)
    i, j = np.triu_indices(k, k=1)
    media_sim = sim_matrix[i, j].mean()
    return float(1 - media_sim)

def microdiversidade(ids, m_gen, m_dir, m_pais, pesos=(1/3, 1/3, 1/3)):
    div_genero = microdiversidade_parametro(m_gen, ids)
    div_diretor = microdiversidade_parametro(m_dir, ids)
    div_pais = microdiversidade_parametro(m_pais, ids)
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
def macrodiversidade_filmes(lista_recomendacoes, matriz):
    todos_ids = []
    for rodada in lista_recomendacoes:
        todos_ids.extend(rodada)

    if len(todos_ids) < 2:
        return np.nan

    features = matriz[todos_ids]
    sim_matrix = cosine_similarity(features)
    i, j = np.triu_indices(len(todos_ids), k=1)
    media_sim = sim_matrix[i, j].mean()
    return float(1 - media_sim)

def macrodiversidade_rodadas(lista_recomendacoes, matriz):
    """
    Mede a diversidade média entre rodadas — ou seja, quão diferentes são
    as rodadas entre si em termos das recomendações que cada uma gerou.
    """
    n = len(lista_recomendacoes)
    if n < 2:
        return np.nan

    diversidades = []
    for i in range(n):
        for j in range(i + 1, n):
            filmes_i = lista_recomendacoes[i]
            filmes_j = lista_recomendacoes[j]
            # Representa cada rodada pelo vetor médio dos filmes recomendados nela
            features_i = matriz[filmes_i].mean(axis=0)
            features_j = matriz[filmes_j].mean(axis=0)

            features_i = np.asarray(features_i).ravel().reshape(1, -1)
            features_j = np.asarray(features_j).ravel().reshape(1, -1)

            sim = cosine_similarity(features_i, features_j)[0, 0]
            diversidades.append(1 - sim)

    return float(np.mean(diversidades))


def macrodiversidade_combinada(lista_recomendacoes, matriz, alpha=0.5):
    """
    Calcula a macrodiversidade combinada:
    - alpha define o peso da diversidade global (0–1)
    - (1 - alpha) define o peso da diversidade entre rodadas
    """
    div_filmes = macrodiversidade_filmes(lista_recomendacoes, matriz)
    div_rodadas = macrodiversidade_rodadas(lista_recomendacoes, matriz)

    # Combina ambas as medidas (ponderadas)
    if np.isnan(div_filmes) and np.isnan(div_rodadas):
        return np.nan
    elif np.isnan(div_filmes):
        return div_rodadas
    elif np.isnan(div_rodadas):
        return div_filmes
    else:
        return float(alpha * div_filmes + (1 - alpha) * div_rodadas)

def macrodiversidade_geral(lista_recomendacoes,m_gen,m_dir,m_pais,pesos=(1 / 3, 1 / 3, 1 / 3),ultimas_x=None,alpha=0.5):
    if ultimas_x is not None:
        lista_recomendacoes = lista_recomendacoes[-ultimas_x:]

    div_genero = macrodiversidade_combinada(lista_recomendacoes, m_gen, alpha)
    div_diretor = macrodiversidade_combinada(lista_recomendacoes, m_dir, alpha)
    div_pais = macrodiversidade_combinada(lista_recomendacoes, m_pais, alpha)

    div_total = (pesos[0] * div_genero +
            pesos[1] * div_diretor +
            pesos[2] * div_pais
    )

    return {
        "macro_div_genero": div_genero,
        "macro_div_diretor": div_diretor,
        "macro_div_pais": div_pais,
        "macro_div_total": div_total
    }