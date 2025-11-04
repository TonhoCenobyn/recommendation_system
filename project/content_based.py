from sklearn.metrics.pairwise import cosine_similarity
from content_based.preferencias import Preferencias, Rodada
from content_based.recomendador import (
    gerar_recomendacao,
    gerar_recomendacao_controlavel)
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from content_based.diversidade import (macrodiversidade_geral,
                                       macrodiversidade_rodadas,
                                       macrodiversidade_filmes,
                                       microdiversidade,
                                       microdiversidade_parametro,
                                       macrodiversidade_combinada,
                                       reajustar_relevancia)
from content_based.usuario import inserir_entradas
from collections import Counter

NUM_RODADAS = 15      # número de rodadas automáticas
TOP_N = 20    # número recomendações por rodada

NIVEL_ALEATORIEDADE_RECOMENDACOES = 0 # (0=0%, 1=20%, 2=50%, 3=80%)

movies = pd.read_csv("dataset/dataset_distribuido.csv")

vectorizer_genero = TfidfVectorizer()
vectorizer_diretor = TfidfVectorizer()
vectorizer_pais = TfidfVectorizer()

matriz_genero = vectorizer_genero.fit_transform(movies["genero"])
matriz_diretor = vectorizer_diretor.fit_transform(movies["diretor"])
matriz_pais = vectorizer_pais.fit_transform(movies["pais_origem"])

def executar_algoritmo(nivel_aleatoriedade_entradas):
    print("-" * 80)
    print("INICIANDO SISTEMA DE RECOMENDAÇÃO")
    print("-" * 80)

    preferencias = Preferencias(alpha=0.1)

    historico_entradas = {
        "genero": [],
        "diretor": [],
        "pais_origem": []
    }

    historico_recomendados_por_rodada = []
    rodadas = []

    for iteracao in range(1, NUM_RODADAS + 1):
        print("-" * 80)
        print(f"RODADA: {iteracao}")
        print("-" * 80)

        genero, diretor, pais = inserir_entradas(movies, historico_entradas, nivel_aleatoriedade_entradas)

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

        #sim_scores, sim_genero, sim_diretor, sim_pais = gerar_recomendacao(preferencias,
        #                                                                   matriz_genero, matriz_diretor, matriz_pais, 1)

        sim_scores, sim_genero, sim_diretor, sim_pais = gerar_recomendacao_controlavel(
            preferencias,
            matriz_genero,
            matriz_diretor,
            matriz_pais,
            NIVEL_ALEATORIEDADE_RECOMENDACOES,
            TOP_N
        )
        ids_recomendados = [i[0] for i in sim_scores[:TOP_N]]
        historico_recomendados_por_rodada.append(ids_recomendados)

        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)

        rodada = Rodada(ids_recomendados)

        print("\nRECOMENDAÇÕES:")
        colunas = ["titulo", "genero", "diretor", "pais_origem"]
        print(movies.iloc[ids_recomendados][colunas].reset_index(drop=True))
        print("-" * 80)

        relevancia_genero_entrada = np.mean(sim_genero_rodada[0, ids_recomendados])
        relevancia_diretor_entrada = np.mean(sim_diretor_rodada[0, ids_recomendados])
        relevancia_pais_entrada = np.mean(sim_pais_rodada[0, ids_recomendados])
        relevancia_media_entrada = float(np.nanmean([relevancia_genero_entrada, relevancia_diretor_entrada, relevancia_pais_entrada]))
        relevancia_media_entrada = reajustar_relevancia(relevancia_media_entrada, referencia=0.7)
        rodada.relevancia_entrada = relevancia_media_entrada

        relevancia_genero_perfil = np.mean(sim_genero[0, ids_recomendados])
        relevancia_diretor_perfil = np.mean(sim_diretor[0, ids_recomendados])
        relevancia_pais_perfil = np.mean(sim_pais[0, ids_recomendados])
        relevancia_media_perfil = float(np.nanmean([relevancia_genero_perfil, relevancia_diretor_perfil, relevancia_pais_perfil]))
        relevancia_media_perfil = reajustar_relevancia(relevancia_media_perfil)
        rodada.relevancia_perfil = relevancia_media_perfil

        print("RELEVÂNCIA:")
        print("\nRelevância considerando o perfil:")
        print(f" - Gênero:  {relevancia_genero_perfil:.2f}")
        print(f" - Diretor: {relevancia_diretor_perfil:.2f}")
        print(f" - País:    {relevancia_pais_perfil:.2f}")
        print(f" - Total (média geral): {relevancia_media_perfil:.2f}")

        print("\nRelevância considerando entrada:")
        print(f" - Gênero:  {relevancia_genero_entrada:.2f}")
        print(f" - Diretor: {relevancia_diretor_entrada:.2f}")
        print(f" - País:    {relevancia_pais_entrada:.2f}")
        print(f" - Total (média geral): {relevancia_media_entrada:.2f}")

        rodada.microdiversidade = microdiversidade(ids_recomendados, matriz_genero, matriz_diretor, matriz_pais)
        #rodada.microdiversidade = reajustar_relevancia(rodada.microdiversidade)

        print("\nMicrodiversidade desta rodada:")
        print(f" - Gênero: {rodada.microdiversidade['div_genero']:.2f}")
        print(f" - Diretor: {rodada.microdiversidade['div_diretor']:.2f}")
        print(f" - Pais: {rodada.microdiversidade['div_pais']:.2f}")
        print(f" - Total (ponderado): {rodada.microdiversidade['div_total']:.2f}")

        rodadas.append(rodada)

    microdiversidade_historico = []
    relevancia_perfil_historico = []
    relevancia_entrada_historico = []

    for rodada in rodadas:
        relevancia_perfil_historico.append(rodada.relevancia_perfil)
        relevancia_entrada_historico.append(rodada.relevancia_entrada)
        microdiversidade_historico.append(rodada.microdiversidade)

    print("\nRELATÓRIO FINAL DE EXECUÇÃO:")

    # --- RELEVANCIA ---
    print("-" * 80)
    print("Relevancia:\n")

    print(f"Média da relevância considerando o perfil: {float(np.nanmean(relevancia_perfil_historico)):.2f}\n")
    print(f"Média da relevância considerando a entrada: {float(np.nanmean(relevancia_entrada_historico)):.2f}")

    # --- MICRODIVERSIDADE ---
    microdiversidade_media_geral = np.mean([d['div_total'] for d in microdiversidade_historico])

    print("-" * 80)
    print("Microdiversidade:\n")
    print(f"Média Geral: {microdiversidade_media_geral:.2f}")

    # MACRODIVERSIDADE
    print("-" * 80)
    print("Macrodiversidade:\n")


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

    # GRAFICOS

    #grafico de relevancia e microdiversidade por rodada
    labels_rodadas = [f"Rodada {i + 1}" for i in range(len(rodadas))]

    valores_microdiv = [d['div_total'] for d in microdiversidade_historico]

    fig_barras = go.Figure()

    fig_barras.add_trace(go.Bar(
        x=labels_rodadas,
        y=valores_microdiv,
        name='Microdiversidade',
        marker_color='forestgreen',
        text=[f"{v:.2f}" for v in valores_microdiv],  # Adiciona texto na barra
        textposition='auto'
    ))
    fig_barras.add_trace(go.Bar(
        x=labels_rodadas,
        y=relevancia_perfil_historico,
        name='Relevância (Perfil)',
        marker_color='darkgreen',
        text=[f"{v:.2f}" for v in relevancia_perfil_historico],
        textposition='auto'
    ))
    fig_barras.add_trace(go.Bar(
        x=labels_rodadas,
        y=relevancia_entrada_historico,
        name='Relevância (Entrada)',
        marker_color='seagreen',
        text=[f"{v:.2f}" for v in relevancia_entrada_historico],
        textposition='auto'
    ))

    fig_barras.update_layout(
        title='Variação de Relevância e Microdiversidade',
        xaxis_title='Rodada',
        yaxis_title='Métrica (0.0 a 1.0)',
        barmode='group',  # Esta é a chave para agrupar as colunas
        yaxis_range=[0, 1.05],  # Força o eixo Y a ir de 0 a 1 (com uma margem)
        template="plotly_white",
        legend_title="Métricas",
        xaxis_tickangle=0  # Mantém os labels das rodadas retos
    )
    fig_barras.show()

    # grafico de frequencia de entradas
    contagens = Counter(preferencias_generos + preferencias_diretores + preferencias_paises)

    # Converter para DataFrame (para plotar)
    df_freq = pd.DataFrame({
        "Entrada": list(contagens.keys()),
        "Frequência": list(contagens.values())
    }).sort_values(by="Frequência", ascending=False)

    # print("\nFREQUENCIA DE ENTRADAS")
    # print(df_freq)

    # Gráfico interativo de barras (Plotly)
    fig = go.Figure(go.Treemap(
        labels=df_freq["Entrada"],
        parents=[""] * len(df_freq),  # Cria um "pai" raiz para todos
        values=df_freq["Frequência"],
        textinfo="label+value+percent root",  # Mostra Label, Valor e % Total
        marker_colorscale='Reds'  # Esquema de cores (tons de vermelho)
    ))

    fig.update_layout(
        title="Proporção das Preferências Inseridas pelo Usuário",
        template="plotly_white"
    )

    # fig.show()  # <-- Descomentei para você ver
    return macro_global_total, macro_rodadas_total


def gerar_grafico_macro(niveis, resultados_filmes, resultados_rodadas):
    labels_x = [f"Nível {n}" for n in niveis]
    textos_filmes = [f"{v:.3f}" for v in resultados_filmes]
    textos_rodadas = [f"{v:.3f}" for v in resultados_rodadas]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=labels_x,
        y=resultados_filmes,
        name='Macrodiv. (Filmes)',
        text=textos_filmes,
        textposition="auto",
        marker_color="darkred"  # Cor 1
    ))

    fig.add_trace(go.Bar(
        x=labels_x,
        y=resultados_rodadas,
        name='Macrodiv. (Rodadas)',
        text=textos_rodadas,
        textposition="auto",
        marker_color="crimson"  # Cor 2
    ))

    fig.update_layout(
        title="Variação de Macrodiversidade",
        xaxis_title="Nível de Aleatoriedade das Entradas",
        yaxis_title="Macrodiversidade (Média)",
        barmode='group',
        template="plotly_white",
        legend_title="Tipo de Macrodiversidade",
        yaxis_range=[0, max(max(resultados_filmes), max(resultados_rodadas)) * 1.15]  # Garante que ambas caibam
    )

    # fig.show()


def main():
    print("-" * 80)
    print("INICIANDO SÉRIE DE SIMULAÇÕES")
    print("-" * 80)

    niveis_para_testar = [3]

    resultados_macro_filmes = []
    resultados_macro_rodadas = []

    for nivel in niveis_para_testar:
        macro_filmes, macro_rodadas = executar_algoritmo(nivel)

        resultados_macro_filmes.append(macro_filmes)
        resultados_macro_rodadas.append(macro_rodadas)

    print("\n" + "=" * 80)
    print("SÉRIE DE SIMULAÇÕES CONCLUÍDA")
    print("Resultados Finais (Nível de Entrada vs. Macrodiversidade):")

    for i in range(len(niveis_para_testar)):
        print(f"  - Nível {niveis_para_testar[i]}:")
        print(f"    - Macro (Filmes):  {resultados_macro_filmes[i]:.2f}")
        print(f"    - Macro (Rodadas): {resultados_macro_rodadas[i]:.2f}")

    print("=" * 80)

    gerar_grafico_macro(niveis_para_testar, resultados_macro_filmes, resultados_macro_rodadas)

if __name__ == "__main__":
    main()
