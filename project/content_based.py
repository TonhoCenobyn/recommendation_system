from sklearn.metrics.pairwise import cosine_similarity
from content_based.preferencias import Preferencias
from content_based.recomendador import gerar_recomendacao
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from recommendation_system.project.content_based.diversidade import macrodiversidade_geral, macrodiversidade_rodadas, macrodiversidade_filmes, microdiversidade, microdiversidade_parametro, macrodiversidade_combinada
from recommendation_system.project.content_based.usuario import inserir_entradas

NUM_RODADAS = 10      # número de rodadas automáticas
TOP_N = 10           # número recomendações por rodada

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

    historico_entradas = {
        "genero": [],
        "diretor": [],
        "pais_origem": []
    }

    historico_recomendados = []
    historico_recomendados_por_rodada = []
    diversidade_rodada_historico = []
    relevancia_rodada_historico = []

    for iteracao in range(1, NUM_RODADAS + 1):
        print("-" * 80)
        print(f"RODADA: {iteracao}")
        print("-" * 80)

        genero, diretor, pais = inserir_entradas(movies, historico_entradas, 3)

        historico_entradas["genero"].append(genero)
        historico_entradas["diretor"].append(diretor)
        historico_entradas["pais_origem"].append(pais)

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

        div_rodada = microdiversidade(ids_recomendados, matriz_genero, matriz_diretor, matriz_pais)
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

    # MACRODIVERSIDADE
    print("\nMacrodiversidade:")
    print("-" * 80)

    # --- MACRODIVERSIDADE GLOBAL (filmes combinados) ---
    macro_global_genero = (macrodiversidade_filmes(historico_recomendados_por_rodada, matriz_genero))
    macro_global_diretor = macrodiversidade_filmes(historico_recomendados_por_rodada, matriz_diretor)
    macro_global_pais = macrodiversidade_filmes(historico_recomendados_por_rodada, matriz_pais)
    macro_global_total = (macro_global_genero + macro_global_diretor + macro_global_pais) / 3

    print("MACRODIVERSIDADE GLOBAL (todos os filmes recomendados):")
    print(f" - Gênero:  {macro_global_genero:.2f}")
    print(f" - Diretor: {macro_global_diretor:.2f}")
    print(f" - País:    {macro_global_pais:.2f}")
    print(f" - Total:   {macro_global_total:.2f}")
    print()

    # --- MACRODIVERSIDADE ENTRE RODADAS ---
    macro_rodadas_genero = macrodiversidade_rodadas(historico_recomendados_por_rodada, matriz_genero)
    macro_rodadas_diretor = macrodiversidade_rodadas(historico_recomendados_por_rodada, matriz_diretor)
    macro_rodadas_pais = macrodiversidade_rodadas(historico_recomendados_por_rodada, matriz_pais)
    macro_rodadas_total = (macro_rodadas_genero + macro_rodadas_diretor + macro_rodadas_pais) / 3

    print("MACRODIVERSIDADE ENTRE RODADAS:")
    print(f" - Gênero:  {macro_rodadas_genero:.2f}")
    print(f" - Diretor: {macro_rodadas_diretor:.2f}")
    print(f" - País:    {macro_rodadas_pais:.2f}")
    print(f" - Total:   {macro_rodadas_total:.2f}")
    print()

    # --- MACRODIVERSIDADE COMBINADA (ponderada por α) ---
    macro_combinada_final = macrodiversidade_geral(
        historico_recomendados_por_rodada,
        matriz_genero, matriz_diretor, matriz_pais,
        alpha=0.5
    )

    print("MACRODIVERSIDADE COMBINADA (α = 0.5):")
    print(f" - Gênero:  {macro_combinada_final['macro_div_genero']:.2f}")
    print(f" - Diretor: {macro_combinada_final['macro_div_diretor']:.2f}")
    print(f" - País:    {macro_combinada_final['macro_div_pais']:.2f}")
    print(f" - Total:   {macro_combinada_final['macro_div_total']:.2f}")
    print("-" * 80)


    preferencias_generos = historico_entradas["genero"]
    preferencias_diretores = historico_entradas["diretor"]
    preferencias_paises = historico_entradas["pais_origem"]

    # Contar frequência de cada item
    from collections import Counter
    contagens = Counter(preferencias_generos + preferencias_diretores + preferencias_paises)

    # Converter para DataFrame (para plotar)
    df_freq = pd.DataFrame({
        "Entrada": list(contagens.keys()),
        "Frequência": list(contagens.values())
    }).sort_values(by="Frequência", ascending=False)

    print("\nFREQUENCIA DE ENTRADAS")
    print(df_freq)

    # Gráfico interativo de barras (Plotly)
    fig = go.Figure(data=[
        go.Bar(
            x=df_freq["Entrada"],
            y=df_freq["Frequência"],
            text=df_freq["Frequência"],
            textposition="auto",
            marker_color="lightskyblue"
        )
    ])
    fig.update_layout(
        title="Frequência das Preferências Inseridas pelo Usuário",
        xaxis_title="Entrada (Gênero / Diretor / País)",
        yaxis_title="Número de vezes escolhido",
        template="plotly_white",
        xaxis_tickangle=45
    )
    fig.show()

if __name__ == "__main__":
    main()
