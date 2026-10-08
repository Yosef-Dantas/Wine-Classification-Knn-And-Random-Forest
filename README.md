# Classificação de Vinhos Tintos e Brancos com KNN e Random Forest

Projeto desenvolvido para a disciplina de Programação Avançada com o objetivo de aplicar técnicas de análise exploratória de dados e Machine Learning em um conjunto de dados relacionado às propriedades físico-químicas de vinhos.

Foram utilizados os algoritmos **K-Nearest Neighbors (KNN)** e **Random Forest**, comparando seus desempenhos por meio de diferentes métricas de classificação, validação cruzada e otimização de hiperparâmetros.

O objetivo do projeto é classificar os vinhos como **tintos (`red`) ou brancos (`white`)** utilizando exclusivamente suas propriedades físico-químicas.

---

## Dataset

O conjunto de dados utilizado foi o **Wine Quality**, disponibilizado pelo UCI Machine Learning Repository:

[Wine Quality Dataset - UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/186/wine+quality)

Foram utilizados os conjuntos referentes aos vinhos tintos e brancos, posteriormente combinados em um único DataFrame.

O dataset original possui **6.497 registros**. Após a identificação e remoção de **1.177 registros duplicados**, permaneceram **5.320 amostras únicas**.

As principais propriedades físico-químicas disponíveis são:

- acidez fixa (`fixed acidity`);
- acidez volátil (`volatile acidity`);
- ácido cítrico (`citric acid`);
- açúcar residual (`residual sugar`);
- cloretos (`chlorides`);
- dióxido de enxofre livre (`free sulfur dioxide`);
- dióxido de enxofre total (`total sulfur dioxide`);
- densidade (`density`);
- pH;
- sulfatos (`sulphates`);
- teor alcoólico (`alcohol`).

O conjunto original também possui a variável `quality`, correspondente à avaliação da qualidade do vinho.

Para este projeto, foi criada a variável:

`wine_type`

responsável por identificar o tipo de vinho:

- `red`: vinho tinto;
- `white`: vinho branco.

A variável `quality` não foi utilizada como característica preditora, pois o objetivo do projeto é classificar o tipo de vinho exclusivamente a partir de suas propriedades físico-químicas.

---

## Análise Exploratória

Durante a análise exploratória foram realizadas etapas como:

- inspeção inicial do conjunto de dados;
- análise dos tipos das variáveis;
- verificação de valores ausentes;
- identificação e remoção de registros duplicados;
- análise da distribuição da variável alvo;
- análise estatística das variáveis numéricas;
- histogramas;
- boxplots;
- comparação das propriedades de acordo com o tipo de vinho;
- análise de correlação;
- levantamento e análise de hipóteses.

Não foram encontrados valores ausentes no conjunto de dados.

Após a remoção dos registros duplicados, a variável `wine_type` apresentou a seguinte distribuição:

- **74,45%** de vinhos brancos;
- **25,55%** de vinhos tintos.

Apesar da diferença entre as classes, não foram aplicadas técnicas de reamostragem. A proporção das classes foi preservada durante a divisão dos dados utilizando estratificação, e o desempenho dos modelos foi analisado também por métricas como precision, recall e F1-score.

---

## Hipóteses Analisadas

Durante a análise exploratória foram investigadas algumas relações entre as propriedades físico-químicas dos vinhos.

### Açúcar residual e densidade

Foi identificada uma associação positiva entre `residual sugar` e `density`.

A correlação geral foi de aproximadamente **0,52**, sendo mais intensa entre os vinhos brancos.

### Dióxido de enxofre livre e total

As variáveis `free sulfur dioxide` e `total sulfur dioxide` apresentaram correlação positiva de aproximadamente **0,72** considerando todas as amostras.

### Acidez fixa e pH

Foi observada uma relação inversa entre `fixed acidity` e `pH`.

A relação negativa mostrou-se mais evidente quando os tipos de vinho foram analisados separadamente.

---

## Pré-processamento

Antes do treinamento dos modelos foram realizadas as seguintes etapas:

- remoção dos registros duplicados;
- definição da variável alvo `wine_type`;
- exclusão das variáveis `wine_type` e `quality` do conjunto de características;
- divisão dos dados em **80% para treinamento e 20% para teste**;
- preservação da proporção das classes utilizando `stratify`;
- padronização das características utilizadas pelo KNN com `StandardScaler`.

Após a divisão:

- **4.256 amostras** foram utilizadas para treinamento;
- **1.064 amostras** foram reservadas para teste.

O `StandardScaler` foi aplicado somente ao KNN, pois o algoritmo utiliza medidas de distância e pode ser influenciado pelas diferentes escalas das variáveis.

O Random Forest não foi padronizado, pois o funcionamento baseado em árvores de decisão não depende da escala dos atributos.

Durante a validação cruzada do KNN, um novo `StandardScaler` foi ajustado exclusivamente sobre a parcela de treinamento de cada fold. Dessa forma, as informações da parcela de validação não participaram do cálculo da média e do desvio-padrão utilizados na padronização, evitando vazamento de dados.

---

## Modelos

### K-Nearest Neighbors

Inicialmente foi utilizado o `KNeighborsClassifier` com sua configuração padrão.

Resultado inicial:

**Acurácia: 99,34%**

Posteriormente foi realizada uma busca de hiperparâmetros utilizando validação cruzada estratificada.

A melhor configuração encontrada foi:

```python
KNeighborsClassifier(
    n_neighbors=11,
    weights="distance"
)
