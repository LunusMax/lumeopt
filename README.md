# ⚽ **LumeOpt — Otimizador de Montagem de Elenco Baseado em Machine Learning e heurística de otimização**

> “Futebol movido por dados, não por impulsos.”

---

## 📘 **Visão Geral**

O **LumeOpt** é um sistema de previsão e otimização de montagem de elencos no futebol profissional.  
Ele utiliza **modelos de aprendizado de máquina**, **redução de dimensionalidade (PCA)** e **heurísticas de busca local** para estimar o desempenho de um time a partir de diferentes combinações de jogadores.

A metodologia é **independente dos dados originais** — qualquer base padronizada de estatísticas de futebol pode ser usada.  
O objeto de registro é o **procedimento algorítmico**, não o dataset.

---

## 🧩 **Objetivo**

- Prever o **desempenho coletivo de uma equipe** (pontuação) a partir de dados de jogadores.  
- Simular **novas contratações** para medir impacto no elenco.  
- Otimizar elencos sob **restrições orçamentárias**, maximizando a performance prevista.

---

## 📂 **Estrutura dos Dados**

Repositório de código e dados: github.com/LunusMax/epl-seriea-data

O LumeOpt opera sobre arquivos `.csv` contendo estatísticas individuais de jogadores por temporada.  
Cada linha representa o desempenho de **um jogador em uma temporada**.

Exemplo de estrutura real (`data_atalanta_2018-2019.csv`):

| Jogador | Posição | Idade | Jogos | Inícios | Minutos | 90s | Gols | Assist. | G+A | Gols s/PK | PK | PK Tent. | Cart. Am. | Cart. Verm. | xG | npxG | xAG | npxG+xAG | Conduções Prog. | Passes Prog. | Receb. Prog. | Gols/90 | Ast/90 | G+A/90 | Gols s/PK/90 | xG/90 | xAG/90 | npxG/90 | npxG+xAG/90 | Div. | Desarmes | 1/3 Def. | 1/3 Central | 1/3 Ataque | Duelo Aéreo | Tentativas | Tkl% | Perdidos | Bloqueios | Tkl+Cortes | Passes | Cortes | Tkl+Int | Defesas | Erros |
|:--------:|:-------:|:-----:|:-----:|:-------:|:--------:|:---:|:----:|:-------:|:---:|:---------:|:--:|:---------:|:----------:|:------------:|:--:|:----:|:----:|:---------:|:---------------:|:-------------:|:-------------:|:-------:|:------:|:-------:|:------------:|:----:|:------:|:-------:|:------------:|:----:|:--------:|:---------:|:-------------:|:-------------:|:-----------:|:-----------:|:----:|:--------:|:----------:|:------------:|:--------:|:--------:|:---------:|:-------:|:------:|
| Nome do jogador | Posição em campo | Idade | Jogos disputados | Jogos como titular | Minutos jogados | 90s jogados | Gols | Assistências | Gols + Assistências | Gols sem pênalti | Pênaltis convertidos | Pênaltis tentados | Cartões amarelos | Cartões vermelhos | Expected Goals | Expected Goals (sem PK) | Expected Assists | Soma (npxG + xAG) | Conduções progressivas | Passes progressivos | Recebimentos progressivos | Gols por 90 min | Assist. por 90 min | G+A por 90 min | Gols s/PK por 90 min | xG/90 | xAG/90 | npxG/90 | npxG+xAG/90 | Divididas | Desarmes ganhos | Ações no terço defensivo | Ações no terço central | Ações no terço ofensivo | Duelos aéreos ganhos | Tentativas de duelo | Taxa de sucesso nos desarmes | Desarmes perdidos | Bloqueios | Tentativas de corte | Precisão de passe | Cortes realizados | Desarmes + Intercepções | Ações defensivas | Erros defensivos |
---

## ⚙️ **Arquitetura do Procedimento**

### 1. **Entrada de Dados**
- Leitura de múltiplos arquivos `.csv` de diferentes ligas e temporadas.
- Seleção automática de colunas relevantes.
- Unificação de nomes de times e filtragem por tempo mínimo de jogo.
- Construção de `df_Players`, contendo todos os jogadores padronizados.
- Valuation de jogadores dentro do df_Players a partir da função que combina **potencial etário** e **desempenho estatístico** dos jogadores.  
  Esse valor é utilizado como base para restrições orçamentárias e otimizações do modelo.

---

### 2. **Redução de Dimensionalidade (PCA)**
- Aplicação de **Principal Component Analysis (PCA)** para condensar as variáveis de desempenho dos jogadores.
- O PCA é **treinado em uma liga-base** (ex: Serie A) e **projetado em outras** (ex: Premier League), criando um **espaço vetorial comparável entre ligas**.
- Cada jogador passa a ser representado por um vetor compacto (`embedding`) que resume seu perfil técnico e tático.

---

### 3. **Treinamento do Modelo**
Modelo neural totalmente conectado com regularização:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l1

model = Sequential([
  Dense(128, activation='relu', kernel_regularizer=l1(0.001)),
  BatchNormalization(),
  Dropout(0.6),
  Dense(32, activation='relu', kernel_regularizer=l1(0.001)),
  BatchNormalization(),
  Dropout(0.6),
  Dense(1)
])
```

- Função de perda: **Mean Squared Error (MSE)**  
- Otimizador: **Adam**  
- Early Stopping configurável  
- Escalonamento do alvo (`StandardScaler`) para estabilidade numérica  

---

### 4. **Predição e Simulação**

Função principal:
```python
preve_pontos(team, season, new_players, player_attributes, player_attributes_PCA, points, budget, df_Players)
```

Processo:
1. Monta o vetor PCA do time alvo com possíveis contratações.
2. Submete o vetor ao modelo para prever o total de pontos esperados.
3. Desfaz a normalização (`inverse_transform`) para valores reais.

---

### 5. **Otimização de Elenco**

Heurística de **busca local**:
```python
busca_local(team, season, player_attributes, player_attributes_PCA, points, budget, df_Players)
```

- Gera uma **solução inicial viável** (respeitando o orçamento).  
- Testa **substituições locais** entre jogadores de diferentes clubes.  
- Avalia o impacto de cada substituição sobre os pontos previstos.  
- Retém apenas soluções com **custo ≤ budget**.  
- Retorna o **melhor conjunto de jogadores** e a **pontuação prevista**.

---

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

---

## 🚀 **Execução**

### 1. Processamento dos dados
```python
df_players = process_league_from_github(folder='Serie_A', league_label='Serie A')
```

### 2. Aplicar PCA e treinar o modelo
```python
df_pca = train_and_apply_pca(df_players)
model = create_model(X_train.shape[1])
model.fit(X_train, y_train, epochs=200, batch_size=16)
```

### 3. Simular novos elencos
```python
pred = preve_pontos(team='Nome_do_time', season='AAAA-AAAA', new_players=[])
```

### 4. Rodar busca local
```python
best_team, best_score = busca_local(
    team='Nome_do_time',
    season='AAAA-AAAA',
    player_attributes=df_all_players,
    player_attributes_PCA=df_all_players_pca,
    points=df_['liga']_points,
    budget='Valor_em_Euros,
    df_Players=df_Players
)
```

---

## 🧠 **Inovação do Procedimento**

- **Generalização entre ligas** via PCA: aprendizado transferível entre competições.  
- Predição **de performance coletiva** a partir de dados individuais.  
- Integração de **Machine Learning + heurística de otimização** sob restrição orçamentária.  
- Método **agnóstico de dados** — aplicável a qualquer base padronizada.  

---

## 🔐 **Registro e Direitos**

Este documento descreve o **procedimento técnico protegido** do projeto **LumeOpt**, desenvolvido por  
**Lucio Vargas de Albuquerque Nunes**, orientado por **Dilson Lucas Pereira**.  

Objeto de registro:
> Método de processamento, modelagem e otimização de elencos futebolísticos baseado em PCA; aprendizado de máquina e heurística de otimização.

Os dados utilizados foram públicos e serviram apenas para validação do método.

---

## 🔢 **Hash de Registro**

```
Algorithm : SHA256
SHA256(LumeOpt.ipynb) = 11CB5DE19A09DCD39F4CDAC450AAFCA5B3C5EC0462939688E7799D762D963A32
```