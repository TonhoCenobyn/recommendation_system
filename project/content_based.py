# content_based/content_based.py
from content_based.preferencias import Preferencias
from content_based.recomendador import gerar_recomendacao
from content_based.diversidade import diversidade_total
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import random
import numpy as np

TOP_N = 30           # número recomendações por rodada
NUM_RODADAS = 5      # número de rodadas automáticas

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

    preferencias = Preferencias(alpha=0.50)
    historico_recomendados = []
    diversidade_rodada_historico = []

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

        preferencias.adicionar(genero, diretor, pais,
                               vectorizer_genero, vectorizer_diretor, vectorizer_pais)

        sim_scores, sim_genero, sim_diretor, sim_pais = gerar_recomendacao(preferencias,
                                                                           matriz_genero, matriz_diretor, matriz_pais)
        ids_recomendados = [i[0] for i in sim_scores[:TOP_N]]
        historico_recomendados.extend(ids_recomendados)

        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        print("\nRECOMENDAÇÕES:")
        colunas = ["titulo", "genero", "diretor", "pais_origem"]
        print(movies.iloc[ids_recomendados][colunas].reset_index(drop=True))
        print("-" * 80)

        relevancia_media_total = np.mean([i[1] for i in sim_scores[:20]])
        relevancia_genero = np.mean(sim_genero[0, ids_recomendados])
        relevancia_diretor = np.mean(sim_diretor[0, ids_recomendados])
        relevancia_pais = np.mean(sim_pais[0, ids_recomendados])

        print("\nRelevância média das recomendações (interesse do usuário):")
        print(f" - Gênero:  {relevancia_genero:.2f}")
        print(f" - Diretor: {relevancia_diretor:.2f}")
        print(f" - País:    {relevancia_pais:.2f}")
        print(f" - Total (média geral): {relevancia_media_total:.2f}")

        div_rodada = diversidade_total(ids_recomendados, matriz_genero, matriz_diretor, matriz_pais)
        diversidade_rodada_historico.append(div_rodada)

        print("\nDiversidade desta rodada:")
        print(f" - Gênero: {div_rodada['div_genero']:.2f}")
        print(f" - Diretor: {div_rodada['div_diretor']:.2f}")
        print(f" - Pais: {div_rodada['div_pais']:.2f}")
        print(f" - Total (ponderado): {div_rodada['div_total']:.2f}")

        if len(diversidade_rodada_historico) >= 5:
            ultimas5 = diversidade_rodada_historico[-5:]
        else:
            ultimas5 = diversidade_rodada_historico

        print("\nMédia de Diversidade das últimas cinco rodadas:")
        if len(ultimas5) > 0:
            print(f" - Gênero: {np.mean([d['div_genero'] for d in ultimas5]):.2f}")
            print(f" - Diretor: {np.mean([d['div_diretor'] for d in ultimas5]):.2f}")
            print(f" - País: {np.mean([d['div_pais'] for d in ultimas5]):.2f}")
            print(f" - Total: {np.mean([d['div_total'] for d in ultimas5]):.2f}")
        else:
            print(" - (sem rodadas suficientes)")

        div_total = diversidade_total(historico_recomendados, matriz_genero, matriz_diretor, matriz_pais)
        print("\nDiversidade acumulada até agora:")
        print(f" - Gênero: {div_total['div_genero']:.2f}")
        print(f" - Diretor: {div_total['div_diretor']:.2f}")
        print(f" - Pais: {div_total['div_pais']:.2f}")
        print(f" - Total (ponderado): {div_total['div_total']:.2f}")

    print("\nRELATÓRIO FINAL DE EXECUÇÃO:")
    print("-" * 80)
    print(f"Diversidade Acumulada final: {div_total['div_total']:.2f}")

    if len(ultimas5) > 0:
        diversidade_ultimas_rodadas = np.mean([d['div_total'] for d in ultimas5])
        print(f"Média de Diversidade das últimas cinco rodadas: {diversidade_ultimas_rodadas:.2f}\n")
        print("CONCLUSÃO: ")
        if diversidade_ultimas_rodadas >= 0.4:
            print("TERMINOU SEM ESTAR EM UMA BOLHA")
        else:
            print("TERMINOU EM UMA BOLHA")
    else:
        print("Poucas rodadas para concluir (menos que 1).")

if __name__ == "__main__":
    main()
