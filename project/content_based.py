# content_based/content_based.py
from sklearn.metrics.pairwise import cosine_similarity
from content_based.preferencias import Preferencias
from content_based.recomendador import gerar_recomendacao
from content_based.diversidade import diversidade_rodada
from content_based.diversidade import macrodiversidade_rodadas
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import plotly.graph_objects as go

NUM_RODADAS = 20      # número de rodadas automáticas
TOP_N = 15           # número recomendações por rodada

movies = pd.read_csv("dataset/dataset_gerado.csv")

vectorizer_genero = TfidfVectorizer()
vectorizer_diretor = TfidfVectorizer()
vectorizer_pais = TfidfVectorizer()

matriz_genero = vectorizer_genero.fit_transform(movies["genero"])
matriz_diretor = vectorizer_diretor.fit_transform(movies["diretor"])
matriz_pais = vectorizer_pais.fit_transform(movies["pais_origem"])

def main():
    print("-" * 80)
    print("INICIANDO SISTEMA DE RECOMENDAÇÃO")
    print("-" * 80)

    preferencias = Preferencias(alpha=0.90)
    historico_recomendados = []
    historico_recomendados_por_rodada = []
    diversidade_rodada_historico = []
    relevancia_rodada_historico = []

    for iteracao in range(1, NUM_RODADAS + 1):
        print("-" * 80)
        print(f"RODADA: {iteracao}")
        print("-" * 80)

        genero = random.choice(movies["genero"].unique())
        diretor = random.choice(movies["diretor"].unique())
        pais = random.choice(movies["pais_origem"].unique())

        print(f"Gênero escolhido: {genero}")
        print(f"Diretor escolhido: {diretor}")
        print(f"País escolhido: {pais}")

        vetor_rodada_genero = vectorizer_genero.transform([genero])
        vetor_rodada_diretor = vectorizer_diretor.transform([diretor])
        vetor_rodada_pais = vectorizer_pais.transform([pais])

        sim_genero_rodada = cosine_similarity(vetor_rodada_genero, matriz_genero)
        sim_diretor_rodada = cosine_similarity(vetor_rodada_diretor, matriz_diretor)
        sim_pais_rodada = cosine_similarity(vetor_rodada_pais, matriz_pais)

        preferencias.adicionar(genero, diretor, pais,
                               vectorizer_genero, vectorizer_diretor, vectorizer_pais)

        sim_scores, sim_genero, sim_diretor, sim_pais = gerar_recomendacao(preferencias,
                                                                           matriz_genero, matriz_diretor, matriz_pais)
        ids_recomendados = [i[0] for i in sim_scores[:TOP_N]]
        historico_recomendados.extend(ids_recomendados)
        historico_recomendados_por_rodada.append(ids_recomendados)

        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        print("\nRECOMENDAÇÕES:")
        colunas = ["titulo", "genero", "diretor", "pais_origem"]
        print(movies.iloc[ids_recomendados][colunas].reset_index(drop=True))
        print("-" * 80)

        relevancia_genero = np.mean(sim_genero_rodada[0, ids_recomendados])
        relevancia_diretor = np.mean(sim_diretor_rodada[0, ids_recomendados])
        relevancia_pais = np.mean(sim_pais_rodada[0, ids_recomendados])
        relevancia_media_total = float(np.nanmean([relevancia_genero, relevancia_diretor, relevancia_pais]))
        relevancia_rodada_historico.append(relevancia_media_total)

        print("\nRelevância média desta rodada:")
        print(f" - Gênero:  {relevancia_genero:.2f}")
        print(f" - Diretor: {relevancia_diretor:.2f}")
        print(f" - País:    {relevancia_pais:.2f}")
        print(f" - Total (média geral): {relevancia_media_total:.2f}")

        div_rodada = diversidade_rodada(ids_recomendados, matriz_genero, matriz_diretor, matriz_pais)
        diversidade_rodada_historico.append(div_rodada)

        print("\nMicrodiversidade desta rodada:")
        print(f" - Gênero: {div_rodada['div_genero']:.2f}")
        print(f" - Diretor: {div_rodada['div_diretor']:.2f}")
        print(f" - Pais: {div_rodada['div_pais']:.2f}")
        print(f" - Total (ponderado): {div_rodada['div_total']:.2f}")

        if len(diversidade_rodada_historico) >= 5:
            ultimas5 = diversidade_rodada_historico[-5:]
        else:
            ultimas5 = diversidade_rodada_historico

    print("\nRELATÓRIO FINAL DE EXECUÇÃO:")
    print("-" * 80)

    microdiversidade_media_geral = np.mean([d['div_total'] for d in diversidade_rodada_historico])
    microdiversidade_ultimas_rodadas = np.mean([d['div_total'] for d in ultimas5])

    print(f"Relevancia média total: {float(np.nanmean(relevancia_rodada_historico)):.2f}\n")
    print("Microdiversidade:")
    print(f" - Média Geral: {microdiversidade_media_geral:.2f}")
    print(f" - Últimas 5 Rodadas: {microdiversidade_ultimas_rodadas:.2f}")


    macro_geral_final = macrodiversidade_rodadas(historico_recomendados_por_rodada,matriz_genero, matriz_diretor, matriz_pais)

    macro_ultimas5_final = macrodiversidade_rodadas(historico_recomendados_por_rodada, matriz_genero, matriz_diretor, matriz_pais, ultimas_x = 5)

    # MACRODIVERSIDADE
    print("\nMacrodiversidade:")
    print(" - Geral:")
    print(f"    Gênero: {macro_geral_final['macro_div_genero']:.2f}")
    print(f"    Diretor: {macro_geral_final['macro_div_diretor']:.2f}")
    print(f"    País: {macro_geral_final['macro_div_pais']:.2f}")
    print(f"    Total: {macro_geral_final['macro_div_total']:.2f}")
    print(" - Últimas 5 rodadas:")
    print(f"    Gênero: {macro_ultimas5_final['macro_div_genero']:.2f}")
    print(f"    Diretor: {macro_ultimas5_final['macro_div_diretor']:.2f}")
    print(f"    País: {macro_ultimas5_final['macro_div_pais']:.2f}")
    print(f"    Total: {macro_ultimas5_final['macro_div_total']:.2f}")

if __name__ == "__main__":
    main()
