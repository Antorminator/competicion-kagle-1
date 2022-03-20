# Competición Kaggle: Titanic - Machine Learning from Disaster

Es este notebook se detalla el desarrollo de un modelo de inteligencia artificial para participar en la competición de Kaggle **["Titanic - Machine Learning from Disaster"](https://www.kaggle.com/c/titanic)**, que reta a predecir la supervivencia de los viajeros en la desastre del Titanic en base a una serie de variables.

Como objetivo propuesto, se intenta que el modelo desarrollado alcance una puntuación superior al **0.7755**.

## Carga de los datos

Como paso inicial, cargamos en un dataframe de Pandas los datos de entrenamiento contenidos en el fichero train.csv, proporcionado en la página de la competición:


```python
import pandas as pd

train_data = pd.read_csv("train.csv")
train_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



Visualizando el dataframe podemos explorar las *features* que contiene nuestro dataset.

Adicionalmente, cargamos también los datos de test que se nos proporcionan, para poder evaluar a priori nuestro modelo antes de mandarlo a puntuar en la plataforma Kaggle:


```python
test_data = pd.read_csv("test.csv")
test_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>3</td>
      <td>Kelly, Mr. James</td>
      <td>male</td>
      <td>34.5</td>
      <td>0</td>
      <td>0</td>
      <td>330911</td>
      <td>7.8292</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>3</td>
      <td>Wilkes, Mrs. James (Ellen Needs)</td>
      <td>female</td>
      <td>47.0</td>
      <td>1</td>
      <td>0</td>
      <td>363272</td>
      <td>7.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>2</td>
      <td>Myles, Mr. Thomas Francis</td>
      <td>male</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>240276</td>
      <td>9.6875</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>3</td>
      <td>Wirz, Mr. Albert</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>315154</td>
      <td>8.6625</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>3</td>
      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>
      <td>female</td>
      <td>22.0</td>
      <td>1</td>
      <td>1</td>
      <td>3101298</td>
      <td>12.2875</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



De la exposición del problema a tratar, y como podemos deducir también al visualizar ambos dataset, nuestra columna objetivo se denominan **Survived**.

## Exploración y tratamiento de los datos


A continuación hacemos una exploración inicial de todo el dataset de entrenamiento, para visualizar las estadisticas de las distintas features:


```python
train_data.info()
train_data.describe()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   PassengerId  891 non-null    int64  
     1   Survived     891 non-null    int64  
     2   Pclass       891 non-null    int64  
     3   Name         891 non-null    object 
     4   Sex          891 non-null    object 
     5   Age          714 non-null    float64
     6   SibSp        891 non-null    int64  
     7   Parch        891 non-null    int64  
     8   Ticket       891 non-null    object 
     9   Fare         891 non-null    float64
     10  Cabin        204 non-null    object 
     11  Embarked     889 non-null    object 
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.7+ KB





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>446.000000</td>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>257.353842</td>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>14.526497</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>223.500000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>20.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>446.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>668.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>891.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>



Tras la exploración inicial, y habiendo revisado la descripción proporcionada por Kaggle de cada columna, se pasa a seleccionar las features consideradas relevantes para el tratamiento del problema:


```python
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

for column in features:
    
    valores_unicos = train_data[column].nunique()
    
    print('Número valores únicos en ',column,': ',valores_unicos)
    if valores_unicos < 10:
        print(train_data[column].unique())
    
    print('Número de NA: ',train_data[column].isna().sum())
```

    Número valores únicos en  Pclass :  3
    [3 1 2]
    Número de NA:  0
    Número valores únicos en  Sex :  2
    ['male' 'female']
    Número de NA:  0
    Número valores únicos en  Age :  88
    Número de NA:  177
    Número valores únicos en  SibSp :  7
    [1 0 3 4 2 5 8]
    Número de NA:  0
    Número valores únicos en  Parch :  7
    [0 1 2 5 3 4 6]
    Número de NA:  0
    Número valores únicos en  Fare :  248
    Número de NA:  0
    Número valores únicos en  Embarked :  3
    ['S' 'C' 'Q' nan]
    Número de NA:  2


Analizando mas detalladamente los valores concretos de cada columna, vemos que en Age y Embarked hay datos faltantes. Registramos la columna Age para darle un tratamiento extra y suprimimos directamente los 2 registros con nan en Embarked:


```python
features_with_na = ["Age"]
train_data.dropna(subset=['Embarked'], inplace=True)
```

## Creación del modelo de aprendizaje automático

En este notebook se utilizará un modelo de tipo **HistGradientBoostingClassifier**, un modelo basado en árboles que utiliza histogramas para acelerar su velocidad de predicción. Se utilizará en un pipeline al que se le proporcionará además los preprocesadores SimpleImputer para las categorías con valores NA y un OrdinalEncoder en general para todas. Se hace uso del preprocesador OrdinalEncoder para evitar la expansión innecesaria del dataset, aprovechando para ello que los modelos basados en árboles no se ven afectados por el orden de los datos.


```python
# Definición de los preprocesadores

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer


ordinal_preprocessor = OrdinalEncoder(handle_unknown="use_encoded_value",
                                       unknown_value=-1)

median_imputer = SimpleImputer(strategy="median")

preprocessor = ColumnTransformer([
    ('median_imputer', median_imputer, features_with_na),
    ('ordinal_preprocessor', ordinal_preprocessor, features)
])

```


```python
# Definición del pipeline con el modelo HistGradientBoostingClassifier

from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", HistGradientBoostingClassifier(random_state=42)),
])
```

Para buscar la mejor configuración del modelo, se hace uso de RandomizedSearchCV para obtener la mejor combinación de hiperparámetros. Para ello, importamos la función loguniform para generar números float aleatorios y creamos otra llamada loguniform_int, que sería homóloga a la anterior pero para números enteros:


```python
from scipy.stats import loguniform


class loguniform_int:
    """Integer valued version of the log-uniform distribution"""
    def __init__(self, a, b):
        self._distribution = loguniform(a, b)

    def rvs(self, *args, **kwargs):
        """Random variable sample"""
        return self._distribution.rvs(*args, **kwargs).astype(int)
```


```python
from sklearn.model_selection import RandomizedSearchCV

# Definición de hiperparámetros a ajustar

param_distributions = {
    'classifier__l2_regularization': loguniform(1e-6, 1e3),
    'classifier__learning_rate': loguniform(0.001, 10),
    'classifier__max_leaf_nodes': loguniform_int(2, 256),
    'classifier__min_samples_leaf': loguniform_int(1, 100),
    'classifier__max_bins': loguniform_int(2, 255),
    'classifier__max_depth': loguniform_int(2, 500)
}

model_random_search = RandomizedSearchCV(
    model, param_distributions=param_distributions, n_iter=20,
    cv=10, verbose=1,
)
```

Para medir la precisión del modelo a entrenar antes de enviar a Kaggle los resultados con los datos de test proporcionados, hacemos una pequeña partición de los datos de entrenamiento para poder contar con nuestros propios datos de test:


```python
from sklearn.model_selection import train_test_split

# Extracción de variable objetivo y variables seleccionadas
y = train_data["Survived"]
X = train_data[features]

# Como el dataset no es muy grande, solo reservamos un 15% de los datos para test

data_train, data_test, target_train, target_test = train_test_split(
    X, y, random_state=42, test_size=0.15) 
```

Y ahora si, realizamos nuestro entrenamiento del modelo usando una búsqueda aleatoria de la mejor combinación de hiperparámetros:


```python
# Búsqueda de hiperparámetros y entrenamiento

model_random_search.fit(data_train, target_train)

# Precisión con datos de test propios

accuracy = model_random_search.score(data_test, target_test)

print(f"Precisión del modelo en datos de test con la combinación de hiperparámetros: "
      f"{accuracy:.4f}")
```

    Fitting 10 folds for each of 20 candidates, totalling 200 fits
    Precisión del modelo en datos de test con la combinación de hiperparámetros: 0.8209


Por último, hacemos las predicciones sobre los datos de test de Kaggle y los guardamos, para hacer el envío a la competición:


```python
predictions = model_random_search.predict(test_data)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Fichero para Kaggle generado correctamente.")
```

    Fichero para Kaggle generado correctamente.
```
