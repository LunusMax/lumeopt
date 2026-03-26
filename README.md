## ⚽ LumeOpt

**What if football transfers were driven by data instead of intuition?**

LumeOpt is a decision-support system that simulates player transfers and predicts their impact on team performance under budget constraints.

It combines:
- Machine Learning (MLP)
- Dimensionality reduction (PCA)
- Optimization heuristics

→ turning raw player data into actionable squad decisions.

------------------------------------------------------------------------

## 🚀 Key Results

- MLP performance: **+55% em relação ao baseline**
- Napoli optimization: **71.93 → 74.66 pontos (+2.73)**
- Cagliari optimization: **41.17 → 42.05 pontos (+0.88)**
- Pipeline completo: **dados → representação → predição → otimização**

------------------------------------------------------------------------

## 📘 Visão Geral

O **LumeOpt** é um sistema de apoio à decisão para montagem de elencos no futebol profissional.

Ele integra:

- **Machine Learning (MLP)** para prever desempenho coletivo  
- **PCA** para representar jogadores em um espaço vetorial comparável  
- **Heurísticas de busca local** para otimizar contratações sob orçamento  

O sistema simula cenários de mercado e estima o impacto de transferências no desempenho de equipes.

> O foco do projeto é o **procedimento algorítmico completo**, não o dataset.

------------------------------------------------------------------------

## 🧠 Pipeline do Sistema

1.  Dados → Jogadores\
2.  Jogadores → Representação (PCA)\
3.  Representação → Desempenho (MLP)\
4.  Predição → Decisão (Otimização)

------------------------------------------------------------------------

## 🎯 Problema Resolvido

Dado um time, uma temporada e um orçamento, o sistema responde:

> **Quais jogadores contratar para maximizar o desempenho esperado?**

------------------------------------------------------------------------

## 🧠 Inovação

-   Predição de performance coletiva a partir de dados individuais\
-   Transferência entre ligas via PCA\
-   Integração de ML + otimização combinatória\
-   Aplicação direta em tomada de decisão no futebol

------------------------------------------------------------------------

## 📂 **Estrutura dos Dados**

Repositório de código e dados: github.com/LunusMax/epl-seriea-data

O LumeOpt opera sobre arquivos `.csv` contendo estatísticas individuais de jogadores por temporada.  
Cada linha representa o desempenho de **um jogador em uma temporada**.

Exemplo de estrutura real (`data_atalanta_2018-2019.csv`):

| Jogador | Posição | Idade | Jogos | Inícios | Minutos | 90s | Gols | Assist. | G+A | Gols s/PK | PK | PK Tent. | Cart. Am. | Cart. Verm. | xG | npxG | xAG | npxG+xAG | Conduções Prog. | Passes Prog. | Receb. Prog. | Gols/90 | Ast/90 | G+A/90 | Gols s/PK/90 | xG/90 | xAG/90 | npxG/90 | npxG+xAG/90 | Div. | Desarmes | 1/3 Def. | 1/3 Central | 1/3 Ataque | Duelo Aéreo | Tentativas | Tkl% | Perdidos | Bloqueios | Tkl+Cortes | Passes | Cortes | Tkl+Int | Defesas | Erros |
|:--------:|:-------:|:-----:|:-----:|:-------:|:--------:|:---:|:----:|:-------:|:---:|:---------:|:--:|:---------:|:----------:|:------------:|:--:|:----:|:----:|:---------:|:---------------:|:-------------:|:-------------:|:-------:|:------:|:-------:|:------------:|:----:|:------:|:-------:|:------------:|:----:|:--------:|:---------:|:-------------:|:-------------:|:-----------:|:-----------:|:----:|:--------:|:----------:|:------------:|:--------:|:--------:|:---------:|:-------:|:------:|
| Nome do jogador | Posição em campo | Idade | Jogos disputados | Jogos como titular | Minutos jogados | 90s jogados | Gols | Assistências | Gols + Assistências | Gols sem pênalti | Pênaltis convertidos | Pênaltis tentados | Cartões amarelos | Cartões vermelhos | Expected Goals | Expected Goals (sem PK) | Expected Assists | Soma (npxG + xAG) | Conduções progressivas | Passes progressivos | Recebimentos progressivos | Gols por 90 min | Assist. por 90 min | G+A por 90 min | Gols s/PK por 90 min | xG/90 | xAG/90 | npxG/90 | npxG+xAG/90 | Divididas | Desarmes ganhos | Ações no terço defensivo | Ações no terço central | Ações no terço ofensivo | Duelos aéreos ganhos | Tentativas de duelo | Taxa de sucesso nos desarmes | Desarmes perdidos | Bloqueios | Tentativas de corte | Precisão de passe | Cortes realizados | Desarmes + Intercepções | Ações defensivas | Erros defensivos |

------------------------------------------------------------------------

## ⚙️ **Arquitetura do Procedimento**

### 1. **Entrada de Dados**
- Leitura de múltiplos arquivos `.csv` de diferentes ligas e temporadas.
- Seleção automática de colunas relevantes.
- Unificação de nomes de times e filtragem por tempo mínimo de jogo.
- Construção de `df_Players`, contendo todos os jogadores padronizados.
- Valuation de jogadores dentro do df_Players a partir da função que combina **potencial etário** e **desempenho estatístico** dos jogadores.  
  Esse valor é utilizado como base para restrições orçamentárias e otimizações do modelo.


### 2. Jogadores → Representação (PCA)

- Redução de dimensionalidade  
- Treinamento em liga base (Serie A)  
- Projeção em outras ligas (Premier League)  
- Criação de **embeddings de jogadores**

👉 Resultado: jogadores representados em um espaço vetorial comparável

------------------------------------------------------------------------

### 3. Representação → Desempenho (MLP)

Modelo neural:

```python
model = Sequential([
  Dense(128, activation='relu', kernel_regularizer=l1(0.001)),
  BatchNormalization(),
  Dropout(0.6),
  Dense(32, activation='relu', kernel_regularizer=l1(0.001)),
  BatchNormalization(),
  Dropout(0.6),
  Dense(1)
])

### 4. **Predição e Simulação**

Função principal:
```python
preve_pontos(team, season, new_players, player_attributes, player_attributes_PCA, points, budget, df_Players)
```

Processo:
1. Monta o vetor PCA do time alvo com possíveis contratações.
2. Submete o vetor ao modelo para prever o total de pontos esperados.
3. Desfaz a normalização (`inverse_transform`) para valores reais.

------------------------------------------------------------------------

### 5. **Otimização de Elenco**

```python
guloso_first_improvement(...)
guloso_best_improvement(...)
busca_local_first_improvement(...)
busca_local_best_improvement(...)

-   Métodos gulosos constroem soluções iniciais de forma incremental,
    selecionando jogadores com base no ganho imediato na função
    objetivo.\
-   Estratégias *first-improvement* aceitam a primeira melhoria
    encontrada, enquanto *best-improvement* avaliam todas as
    alternativas locais antes de decidir.\
-   Métodos de busca local refinam soluções iniciais por meio de
    substituições iterativas no elenco.\
-   Todas as abordagens respeitam restrições orçamentárias e operam
    sobre a mesma função objetivo (pontuação prevista).\
-   Retornam o conjunto de jogadores otimizado e o desempenho esperado
    associado.

------------------------------------------------------------------------

## 🧮 **Dependências**

| Biblioteca | Versão mínima | Função principal |
|-------------|---------------|------------------|
| Python | 3.10 | Linguagem base |
| pandas | 2.0 | Manipulação de dados |
| numpy | 1.26 | Operações numéricas |
| scikit-learn | 1.3 | PCA, Scaler |
| tensorflow | 2.17 | Rede neural |
| matplotlib | 3.7 | Visualizações |

Instalação:
```bash
pip install -r requirements.txt
```

## 🧠 **Inovação do Procedimento**

- **Generalização entre ligas** via PCA: aprendizado transferível entre competições.  
- Predição **de performance coletiva** a partir de dados individuais.  
- Integração de **Machine Learning + heurística de otimização** sob restrição orçamentária.  
- Método **agnóstico de dados** — aplicável a qualquer base padronizada.  

------------------------------------------------------------------------

## 🔐 Intellectual Property & Usage

This repository contains a **public demonstration** of the LumeOpt system.

The full implementation and core optimization logic are intentionally abstracted.

### License
This project is licensed under **Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)**.

You are free to:
- Use and study the code
- Share with attribution

You are NOT allowed to:
- Submit this work as your own
- Use it for commercial purposes without permission

### Authorship

All methods, models and system design are original work developed by:

**Lucio Vargas de Albuquerque Nunes** and **Dilson Lucas Pereira**

Any unauthorized academic submission or reproduction is considered misconduct.

------------------------------------------------------------------------

## 🔢 **Hash de Registro**

```
Algorithm : SHA256
SHA256(LumeOpt.ipynb) = C2BA4DC6E9BF58D5AB7563EBF1DDC56147052109D2B49BA970ABE5AD5498676F
```