"""LLM powered agent capable of executing natural language analysis requests."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
from typing import Any, Dict, Optional

import pandas as pd

from app.analysis.data_loader import dataframe_statistics

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency fallback
    OpenAI = None  # type: ignore


@dataclass
class AgentResponse:
    """Represents the result returned by the analysis agent."""

    explanation: str
    result_table: Optional[pd.DataFrame] = None
    raw_response: Optional[str] = None


class BaseAnalysisAgent:
    """Defines the interface for analysis agents."""

    def analyze(self, prompt: str, dataframe: pd.DataFrame, context: Dict[str, Any]) -> AgentResponse:
        raise NotImplementedError


class LocalPandasAgent(BaseAnalysisAgent):
    """Fallback agent that performs keyword driven analysis using pandas."""

    def analyze(self, prompt: str, dataframe: pd.DataFrame, context: Dict[str, Any]) -> AgentResponse:
        cleaned_prompt = prompt.lower()

        if "missing" in cleaned_prompt or "nulo" in cleaned_prompt:
            missing = dataframe.isna().sum().to_frame(name="missing")
            return AgentResponse(
                explanation="Contagem de valores ausentes por coluna.",
                result_table=missing,
            )

        if "correlation" in cleaned_prompt or "correlação" in cleaned_prompt:
            corr = dataframe.corr(numeric_only=True)
            return AgentResponse(
                explanation="Matriz de correlação para colunas numéricas.",
                result_table=corr,
            )

        if any(keyword in cleaned_prompt for keyword in ["summary", "resumo", "describe", "estatística"]):
            summary = dataframe.describe(include="all").transpose()
            return AgentResponse(
                explanation="Resumo estatístico das colunas.",
                result_table=summary,
            )

        column_match = self._find_column_in_prompt(cleaned_prompt, dataframe.columns)
        if column_match is not None:
            series = dataframe[column_match]
            if pd.api.types.is_numeric_dtype(series):
                desc = series.describe().to_frame(name=column_match)
                return AgentResponse(
                    explanation=f"Estatísticas descritivas para a coluna '{column_match}'.",
                    result_table=desc,
                )
            value_counts = series.value_counts().to_frame(name="count")
            return AgentResponse(
                explanation=f"Distribuição de valores para a coluna '{column_match}'.",
                result_table=value_counts,
            )

        basic_stats = dataframe_statistics(dataframe)
        return AgentResponse(
            explanation=(
                "Agente local não encontrou uma ação específica. "
                "Exibindo estatísticas gerais do conjunto de dados."
            ),
            result_table=pd.DataFrame(basic_stats["numeric_summary"]).transpose()
            if basic_stats["numeric_summary"]
            else None,
            raw_response=json.dumps(basic_stats, ensure_ascii=False, indent=2),
        )

    @staticmethod
    def _find_column_in_prompt(prompt: str, columns: pd.Index) -> Optional[str]:
        for column in columns:
            pattern = re.escape(column.lower())
            if re.search(rf"\b{pattern}\b", prompt):
                return column
        return None


class OpenAIAgent(BaseAnalysisAgent):
    """Agent powered by OpenAI's responses."""

    def __init__(self, *, model: str = "gpt-4o-mini", temperature: float = 0.2) -> None:
        if OpenAI is None:
            raise ImportError(
                "O pacote 'openai' não está instalado. Instale com `pip install openai`."
            )
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "Defina a variável de ambiente OPENAI_API_KEY para utilizar o agente OpenAI."
            )
        self._client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def analyze(self, prompt: str, dataframe: pd.DataFrame, context: Dict[str, Any]) -> AgentResponse:
        preview = dataframe.head(20).to_dict(orient="records")
        payload = {
            "schema": context,
            "preview": preview,
        }

        system_prompt = (
            "Você é um assistente de dados. Utilize as informações fornecidas para responder ao usuário. "
            "Retorne resultados em Markdown e indique claramente passos de análise e conclusões."
        )

        response = self._client.responses.create(
            model=self.model,
            temperature=self.temperature,
            input=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "input_json",
                            "input_json": payload,
                        },
                    ],
                },
            ],
        )

        text_response = response.output[0].content[0].text if response.output else ""
        return AgentResponse(explanation=text_response, raw_response=text_response)


class AgentFactory:
    """Utility factory that returns the most capable available agent."""

    @staticmethod
    def create() -> BaseAnalysisAgent:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and OpenAI is not None:
            try:
                return OpenAIAgent()
            except Exception:
                pass
        return LocalPandasAgent()
