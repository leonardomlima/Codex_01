import pandas as pd
import pytest

from app.agent.llm_agent import AgentFactory, LocalPandasAgent


def _sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sales": [100, 150, 200, None],
            "region": ["sul", "sudeste", "sul", "norte"],
        }
    )


def test_local_agent_missing_values():
    agent = LocalPandasAgent()
    df = _sample_dataframe()
    response = agent.analyze("Quais colunas possuem valores nulos?", df, context={})

    assert "ausentes" in response.explanation.lower()
    assert response.result_table is not None
    assert response.result_table.loc["sales", "missing"] == 1


def test_local_agent_column_detection_numeric():
    agent = LocalPandasAgent()
    df = _sample_dataframe()
    response = agent.analyze("Me mostre estatísticas da coluna sales", df, context={})

    assert "sales" in response.explanation.lower()
    assert "mean" in response.result_table.index


def test_local_agent_default_statistics():
    agent = LocalPandasAgent()
    df = _sample_dataframe()
    response = agent.analyze("conte-me algo", df, context={})

    assert "estatísticas gerais" in response.explanation.lower()
    assert response.raw_response is not None


@pytest.mark.parametrize("env_key", [None, ""])
def test_agent_factory_without_openai(monkeypatch: pytest.MonkeyPatch, env_key: str | None):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    if env_key is not None:
        monkeypatch.setenv("OPENAI_API_KEY", env_key)

    # Garantir que a importação opcional não force dependência externa nos testes
    monkeypatch.setattr("app.agent.llm_agent.OpenAI", None)

    agent = AgentFactory.create()
    assert isinstance(agent, LocalPandasAgent)
