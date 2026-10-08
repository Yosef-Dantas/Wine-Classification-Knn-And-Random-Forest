# Classificação de Vinhos Tintos e Brancos com KNN e Random Forest

Projeto desenvolvido para a disciplina de **Programação Avançada** com o objetivo de aplicar técnicas de **Análise Exploratória de Dados (EDA)** e **Machine Learning** ao conjunto de dados **Wine Quality**.

Foram utilizados os algoritmos **K-Nearest Neighbors (KNN)** e **Random Forest**, comparando seus desempenhos por meio de métricas de classificação, validação cruzada e otimização de hiperparâmetros.

O objetivo principal do projeto é classificar os vinhos como **tintos (`red`)** ou **brancos (`white`)** utilizando suas propriedades físico-químicas.

---

## Dataset

O conjunto de dados utilizado foi o **Wine Quality**, disponibilizado pelo **UCI Machine Learning Repository**:

[Wine Quality Dataset - UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/186/wine+quality)

Foram utilizados os conjuntos referentes aos vinhos tintos e brancos, posteriormente combinados em um único DataFrame.

O conjunto original possui **6.497 registros**. Após a identificação e remoção de **1.177 registros duplicados**, permaneceram **5.320 amostras únicas**.

As propriedades físico-químicas utilizadas no projeto são:

- acidez fixa (`fixed acidity`);
- acidez volátil (`volatile acidity`);
- ácido cítrico (`citric acid`);
- açúcar residual (`residual sugar`);
- cloretos (`chlorides`);
- dióxido de enxofre livre (`free sulfur dioxide`);
- dióxido de enxofre total (`total sulfur dioxide`);
- densidade (`density`);
- pH (`pH`);
- sulfatos (`sulphates`);
- teor alcoólico (`alcohol`).

O conjunto original também possui a variável `quality`, correspondente à avaliação da qualidade do vinho.

Para este projeto foi criada a variável-alvo:

```text
wine_type
```

responsável por identificar o tipo de vinho:

- `red`: vinho tinto;
- `white`: vinho branco.

A variável `quality` não foi utilizada como característica preditora, pois o objetivo do projeto é classificar o **tipo de vinho exclusivamente a partir de suas propriedades físico-químicas**.

---

## Análise Exploratória de Dados

Durante a análise exploratória foram realizadas as seguintes etapas:

- inspeção inicial do conjunto de dados;
- análise dos tipos das variáveis;
- verificação de valores ausentes;
- identificação e remoção de registros duplicados;
- análise da distribuição da variável-alvo;
- análise estatística das variáveis numéricas;
- histogramas;
- boxplots;
- comparação das propriedades físico-químicas entre vinhos tintos e brancos;
- análise de correlação;
- levantamento e análise de hipóteses.

Não foram encontrados valores ausentes no conjunto de dados.

Após a remoção dos registros duplicados, a variável `wine_type` apresentou a seguinte distribuição:

- **74,45%** de vinhos brancos;
- **25,55%** de vinhos tintos.

Apesar da diferença entre as classes, não foram aplicadas técnicas de reamostragem. A proporção das classes foi preservada na divisão entre treinamento e teste utilizando `stratify`, e o desempenho dos modelos também foi analisado por métricas específicas de cada classe, como **precision**, **recall** e **F1-score**.

---

## Hipóteses Analisadas

Durante a análise exploratória foram investigadas relações entre algumas propriedades físico-químicas dos vinhos.

### Açúcar residual e densidade

Foi observada uma associação positiva entre `residual sugar` e `density`.

A correlação geral foi de aproximadamente **0,52**, sendo mais intensa entre os vinhos brancos.

### Dióxido de enxofre livre e total

As variáveis `free sulfur dioxide` e `total sulfur dioxide` apresentaram correlação positiva de aproximadamente **0,72** considerando todas as amostras.

### Acidez fixa e pH

Foi observada uma relação inversa entre `fixed acidity` e `pH`.

A associação negativa mostrou-se mais evidente quando os tipos de vinho foram analisados separadamente.

---

## Pré-processamento

Antes do treinamento dos modelos foram realizadas as seguintes etapas:

- remoção dos registros duplicados;
- definição da variável-alvo `wine_type`;
- exclusão de `wine_type` e `quality` do conjunto de características;
- divisão dos dados em **80% para treinamento e 20% para teste**;
- manutenção da proporção das classes utilizando `stratify`;
- padronização das características utilizadas pelo KNN com `StandardScaler`.

Após a divisão:

- **4.256 amostras** foram utilizadas para treinamento;
- **1.064 amostras** foram utilizadas para teste.

O `StandardScaler` foi aplicado somente ao KNN, pois esse algoritmo utiliza medidas de distância e pode ser influenciado por diferenças de escala entre as variáveis.

O Random Forest não foi padronizado, pois seu funcionamento é baseado em árvores de decisão e não depende da escala dos atributos.

Durante a validação cruzada do KNN, um novo `StandardScaler` foi ajustado exclusivamente sobre a parcela de treinamento de cada fold. Dessa forma, as informações da parcela de validação não participaram do cálculo da média e do desvio-padrão utilizados na padronização, evitando vazamento de dados.

---

## Modelos

### K-Nearest Neighbors

Inicialmente foi utilizado o `KNeighborsClassifier` com sua configuração padrão.

Resultado inicial:

**Acurácia: 99,34%**

Posteriormente foi realizada uma busca manual de hiperparâmetros utilizando validação cruzada estratificada.

A melhor configuração encontrada foi:

```python
KNeighborsClassifier(
    n_neighbors=11,
    weights="distance"
)
```

O melhor F1-macro médio durante a validação cruzada foi:

**98,92%**

Na validação cruzada com cinco divisões, o modelo apresentou:

**Acurácia média: 99,18%**

Após o treinamento do modelo final com os hiperparâmetros selecionados:

**Acurácia no conjunto de teste: 99,25%**

A matriz de confusão apresentou:

- 268 vinhos tintos classificados corretamente;
- 788 vinhos brancos classificados corretamente;
- 8 classificações incorretas.

---

### Random Forest

Inicialmente foi utilizado o `RandomForestClassifier` com `random_state=42`.

Resultado inicial:

**Acurácia: 99,44%**

Posteriormente foi utilizado o `GridSearchCV` para investigar diferentes combinações de hiperparâmetros.

A melhor configuração encontrada foi:

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=2,
    random_state=42
)
```

O melhor F1-macro médio durante a validação cruzada foi:

**99,19%**

Na validação cruzada com cinco divisões, o modelo apresentou:

**Acurácia média: 99,39%**

No conjunto de teste:

**Acurácia: 99,44%**

A matriz de confusão apresentou:

- 267 vinhos tintos classificados corretamente;
- 791 vinhos brancos classificados corretamente;
- 6 classificações incorretas.

---

## Comparação dos Modelos

| Modelo | Acurácia Inicial | Acurácia Final | Acurácia Média CV | F1-Macro CV |
| --- | ---: | ---: | ---: | ---: |
| KNN | 99,34% | 99,25% | 99,18% | 98,92% |
| Random Forest | 99,44% | 99,44% | 99,39% | 99,19% |

Os dois modelos apresentaram desempenho muito superior ao requisito mínimo de **85% de acurácia** estabelecido para o projeto.

O **Random Forest apresentou desempenho ligeiramente superior e mais consistente**, tanto no conjunto de teste quanto durante a validação cruzada.

Por esse motivo, o Random Forest foi escolhido como modelo final.

---

## Importância das Características

A análise de importância das variáveis do Random Forest mostrou que as características com maior contribuição para a classificação foram:

1. dióxido de enxofre total (`total sulfur dioxide`);
2. cloretos (`chlorides`);
3. acidez volátil (`volatile acidity`);
4. densidade (`density`);
5. sulfatos (`sulphates`);
6. dióxido de enxofre livre (`free sulfur dioxide`).

Entre as variáveis com menor importância relativa estão:

- pH;
- ácido cítrico (`citric acid`);
- teor alcoólico (`alcohol`).

A importância atribuída pelo Random Forest representa a contribuição das características para as decisões realizadas pelas árvores e **não deve ser interpretada como uma relação de causa e efeito**.

---

## Modelo Salvo

O melhor modelo foi salvo utilizando a biblioteca `joblib`:

```python
melhor_modelo = grid_rf.best_estimator_

joblib.dump(
    melhor_modelo,
    "modelo_random_forest_wine.joblib"
)
```

Após o salvamento, o modelo foi carregado novamente para verificar se seu desempenho havia sido preservado:

```python
modelo_carregado = joblib.load(
    "modelo_random_forest_wine.joblib"
)

previsoes_teste = modelo_carregado.predict(X_test)
```

O modelo carregado manteve:

**Acurácia: 99,44%**

confirmando que o processo de armazenamento preservou corretamente o estimador treinado.

---

## Arquivos do Repositório

```text
.
├── Classificacao_Vinhos_KNN_RandomForest.ipynb
├── modelo_random_forest_wine.joblib
└── README.md
```

---

## Tecnologias Utilizadas

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- Jupyter Notebook / Google Colab

---

## Principais Recursos do Scikit-learn

Foram utilizados recursos como:

- `train_test_split`;
- `StratifiedKFold`;
- `GridSearchCV`;
- `cross_val_score`;
- `StandardScaler`;
- `KNeighborsClassifier`;
- `RandomForestClassifier`;
- `accuracy_score`;
- `f1_score`;
- `confusion_matrix`;
- `ConfusionMatrixDisplay`;
- `classification_report`.

---

## Execução

O projeto pode ser executado utilizando **Jupyter Notebook** ou **Google Colab**.

As principais dependências podem ser instaladas com:

```bash
pip install pandas numpy matplotlib scikit-learn joblib
```

Depois, basta abrir e executar o notebook:

```text
Classificacao_Vinhos_KNN_RandomForest.ipynb
```

Os dados são carregados diretamente do **UCI Machine Learning Repository** durante a execução do notebook, portanto não é necessário armazenar os arquivos CSV manualmente no repositório.

---

## Conclusão

A análise mostrou que as propriedades físico-químicas disponíveis no conjunto Wine Quality permitem distinguir vinhos tintos e brancos com elevado desempenho.

O KNN apresentou **99,25% de acurácia no modelo final**, enquanto o Random Forest alcançou **99,44%**.

Durante a validação cruzada, as acurácias médias foram de **99,18% para o KNN** e **99,39% para o Random Forest**. Os valores de F1-macro obtidos durante a otimização também favoreceram ligeiramente o Random Forest, com **99,19%**, em comparação aos **98,92%** obtidos pelo KNN.

Além de apresentar o melhor desempenho geral, o Random Forest permitiu analisar a importância das propriedades físico-químicas utilizadas na classificação.

Com base nesses resultados, o **Random Forest foi selecionado como modelo final** e armazenado utilizando Joblib para possibilitar sua reutilização posterior.
