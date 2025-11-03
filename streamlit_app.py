"""Streamlit front-end for the intelligent data analysis platform."""
from __future__ import annotations

import streamlit as st

from app.analysis.data_loader import DatasetMetadata, generate_metadata, load_dataset
from app.analysis.analyzer import DataAnalyzer


st.set_page_config(page_title="Data Copilot", layout="wide")


def _render_metadata(metadata: DatasetMetadata) -> None:
    st.subheader("Resumo do dataset")
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    metrics_col1.metric("Linhas", metadata.rows)
    metrics_col2.metric("Colunas", metadata.columns)
    metrics_col3.metric("Arquivo", metadata.filename)

    with st.expander("Pré-visualização", expanded=True):
        st.dataframe(metadata.preview)

    with st.expander("Tipos de dados"):
        st.json(metadata.dtypes)

    with st.expander("Valores ausentes"):
        st.json(metadata.missing_values)

    if metadata.categorical_summaries:
        with st.expander("Distribuições categóricas"):
            st.json(metadata.categorical_summaries)

    if metadata.numerical_summaries:
        with st.expander("Estatísticas numéricas"):
            st.json(metadata.numerical_summaries)


def main() -> None:
    st.title("Copiloto de Análise de Dados")
    st.write(
        "Faça o upload de um arquivo CSV ou XLSX, explore os dados e descreva a análise desejada em linguagem natural."
    )

    uploaded_file = st.file_uploader("Selecione um arquivo", type=["csv", "xlsx"])
    prompt = st.text_area("Descreva a análise desejada", placeholder="Ex: Mostre a média da coluna de faturamento por mês.")

    if uploaded_file is None:
        st.info("Carregue um arquivo para começar.")
        return

    dataframe = load_dataset(uploaded_file, uploaded_file.name)
    metadata = generate_metadata(dataframe, uploaded_file.name)

    _render_metadata(metadata)

    run_button = st.button("Executar análise", disabled=not prompt)

    if run_button and prompt:
        with st.spinner("Consultando agente..."):
            analyzer = DataAnalyzer(dataframe, metadata)
            result = analyzer.analyze_prompt(prompt)

        response = result.response
        st.markdown("### Resultado do agente")
        st.markdown(response.explanation)

        if response.result_table is not None:
            st.markdown("#### Saída tabular")
            st.dataframe(response.result_table)

        if response.raw_response and response.result_table is None:
            with st.expander("Resposta detalhada"):
                st.code(response.raw_response, language="json")


if __name__ == "__main__":
    main()
