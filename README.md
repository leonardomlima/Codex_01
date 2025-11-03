# Copiloto de Análise de Dados

Aplicação web em Python para análise exploratória de arquivos CSV/XLSX com um agente de IA integrado. O front-end utiliza Streamlit para oferecer uma interface amigável e responsiva.

## Funcionalidades

- Upload de arquivos `.csv` ou `.xlsx`.
- Perfil automático do dataset (tipos de dados, valores ausentes, estatísticas básicas e pré-visualização).
- Agente inteligente capaz de interpretar prompts em linguagem natural e executar análises:
  - Agente OpenAI (quando `OPENAI_API_KEY` está configurada).
  - Agente local baseado em pandas como fallback para ambientes offline.
- Exibição de resultados textuais e tabulares diretamente na interface.

## Requisitos

- Python 3.10+
- Dependências listadas em `requirements.txt`
- (Opcional) Chave de API da OpenAI para utilizar o agente LLM

## Como executar

1. Crie um ambiente virtual e instale as dependências:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
   pip install -r requirements.txt
   ```

2. (Opcional) Defina a variável de ambiente com a chave da OpenAI:

   ```bash
   export OPENAI_API_KEY="sua-chave"  # Windows PowerShell: $env:OPENAI_API_KEY="sua-chave"
   ```

3. Inicie a aplicação Streamlit:

   ```bash
   streamlit run streamlit_app.py
   ```

4. Acesse o endereço exibido no terminal, faça upload do arquivo e interaja com o agente via prompt em linguagem natural.

## Como testar

Após instalar as dependências, execute a suíte de testes automatizados para validar as principais utilidades do projeto:

```bash
pytest
```

## Estrutura do projeto

```
app/
  agent/
    llm_agent.py         # Implementação do agente OpenAI e fallback local
  analysis/
    data_loader.py       # Utilidades para leitura e perfilamento de datasets
    analyzer.py          # Orquestra o fluxo entre dataset e agente
streamlit_app.py         # Front-end Streamlit
requirements.txt         # Dependências do projeto
```

## Observações

- O agente local oferece respostas baseadas em regras simples para permitir testes rápidos sem dependências externas.
- Para melhores resultados em análises complexas, utilize a integração com a OpenAI.
