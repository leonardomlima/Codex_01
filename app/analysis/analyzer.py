"""High-level routines for orchestrating data analysis."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from app.agent.llm_agent import AgentFactory, AgentResponse
from app.analysis.data_loader import DatasetMetadata


@dataclass
class AnalysisResult:
    """Represents the structured output of a natural language analysis request."""

    response: AgentResponse
    metadata: DatasetMetadata


class DataAnalyzer:
    """Coordinates the flow between the dataset metadata and the chosen agent."""

    def __init__(self, dataframe: pd.DataFrame, metadata: DatasetMetadata):
        self.dataframe = dataframe
        self.metadata = metadata
        self.agent = AgentFactory.create()

    def analyze_prompt(self, prompt: str) -> AnalysisResult:
        context = {
            "dtypes": self.metadata.dtypes,
            "missing_values": self.metadata.missing_values,
            "categorical_summaries": self.metadata.categorical_summaries,
            "numerical_summaries": self.metadata.numerical_summaries,
            "rows": self.metadata.rows,
            "columns": self.metadata.columns,
        }
        response = self.agent.analyze(prompt, self.dataframe, context)
        return AnalysisResult(response=response, metadata=self.metadata)
