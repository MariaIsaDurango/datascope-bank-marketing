<div align="center">

# 🏦 Predicción de Suscripción a Depósitos a Plazo
### DataScope Solutions — Proyecto de Machine Learning

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?style=flat&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-green?style=flat&logo=pandas)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat&logo=jupyter)
![Estado](https://img.shields.io/badge/Estado-Completado-brightgreen?style=flat)
![Metodología](https://img.shields.io/badge/Metodología-CRISP--DM-blue?style=flat)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.811-success?style=flat)

</div>

---

## 📋 Descripción del proyecto

Este proyecto forma parte del portafolio profesional de **DataScope Solutions**,
una consultora ficticia de Ciencia de Datos.

El objetivo es desarrollar un modelo de Machine Learning capaz de predecir si un
cliente de una institución bancaria contratará un **depósito a plazo fijo** tras
ser contactado por una campaña de telemarketing.

Predecir correctamente qué clientes están dispuestos a suscribirse permite al banco
**optimizar sus campañas de marketing**, reducir costos operativos y aumentar la
tasa de conversión — sin necesidad de contactar a clientes que rechazarán la oferta.

---

## 🎯 Problema de negocio

Las campañas de telemarketing bancario tienen un costo económico y de tiempo
significativo. Contactar a todos los clientes disponibles no es eficiente, dado
que la mayoría no aceptará la oferta.

> **¿Podemos predecir, antes de realizar la llamada, qué clientes tienen mayor
> probabilidad de suscribirse a un depósito a plazo?**

---

## 🔍 Tipo de problema

| Característica | Detalle |
|---|---|
| **Tipo de ML** | Aprendizaje supervisado |
| **Categoría** | Clasificación binaria |
| **Variable objetivo** | `y` → yes / no |
| **Desbalanceo de clases** | ~11% positivos (yes) / ~89% negativos (no) |
| **Metodología** | CRISP-DM |

---

## 📊 Dataset

| Característica | Detalle |
|---|---|
| **Nombre** | Bank Marketing Dataset |
| **Fuente** | UCI Machine Learning Repository |
| **Referencia** | Moro et al. (2014) — *A Data-Driven Approach to Predict the Success of Bank Telemarketing* |
| **Registros** | 41.188 clientes |
| **Variables** | 20 predictoras + 1 target |
| **Período** | Campañas 2008–2013 (banco portugués) |
| **Acceso** | [UCI Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing) |

---

## 🏆 Resultados del Modelo

El modelo seleccionado fue **Random Forest**, entrenado con 32.950 muestras
y evaluado sobre 8.238 muestras nunca vistas durante el entrenamiento.

### Métricas en Test (datos nunca vistos)

| Métrica | Clase `yes` | Interpretación |
|---|---|---|
| **Recall** | **62.5%** | De cada 100 clientes interesados, el modelo identifica 62 |
| **Precision** | **44.8%** | De cada 100 llamadas recomendadas, 45 convierten |
| **F1-Score** | **52.2%** | Equilibrio entre Recall y Precision |
| **ROC-AUC** | **81.1%** | El modelo discrimina positivos de negativos el 81% de las veces |
| **Accuracy global** | **87.1%** | — |

### Comparación con Baseline

| Modelo | F1 (yes) | Recall (yes) | ROC-AUC |
|---|---|---|---|
| Baseline (siempre predice "no") | 0.000 | 0.000 | 0.500 |
| Logistic Regression | — | — | — |
| **Random Forest** ✅ | **0.522** | **0.625** | **0.811** |
| Gradient Boosting | — | — | — |

### Impacto Operativo Estimado

| Escenario | Llamadas necesarias | Clientes positivos capturados |
|---|---|---|
| **Sin modelo** (llamar a todos) | 8.238 | 928 (100%) |
| **Con modelo** (Random Forest) | ~1.295 | 580 (62.5%) |
| **Reducción** | **~84% menos llamadas** | 62.5% de conversión |

> El modelo permite una **eficiencia de conversión 4 veces superior** a llamar
> aleatoriamente, con una reducción estimada del 84% en costes de llamadas.

### Variables más Predictivas

| Rank | Variable | Importancia | Interpretación de negocio |
|---|---|---|---|
| 🥇 1 | `euribor3m` | Alta | El contexto económico es el factor más determinante |
| 🥈 2 | `nr.employed` | Alta | El ciclo económico captura el momento óptimo de campaña |
| 🥉 3 | `poutcome_success` | Alta | Clientes con éxito previo: 5x más propensos a suscribir |
| 4 | `age` | Media | Mayor edad → mayor propensión al ahorro conservador |
| 5 | `campaign_capped` | Media | Menos llamadas necesarias = mejor candidato |

---

## 🛠️ Tecnologías utilizadas

| Categoría | Herramientas |
|---|---|
| **Lenguaje** | Python 3.9+ |
| **Manipulación de datos** | Pandas, NumPy |
| **Visualización** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn |
| **Entorno** | Jupyter Notebook |
| **Control de versiones** | Git, GitHub |
| **Serialización del modelo** | Joblib |
| **Metodología** | CRISP-DM |

---

## 📁 Estructura del proyecto

```
datascope-bank-marketing/
│
├── data/
│   ├── raw/                    <- Datos originales sin modificar
│   └── processed/              <- Datos limpios listos para modelar
│
├── notebooks/
│   └── bank_marketing_ml_project.ipynb   <- Notebook principal
│
├── models/
│   └── best_model_random_forest.pkl      <- Modelo entrenado (joblib)
│
├── reports/
│   └── figures/                <- Visualizaciones generadas
│       ├── 01_target_distribution.png
│       ├── 02_numerical_distributions.png
│       ├── 03_categorical_distributions.png
│       ├── 04_conversion_rate_categorical.png
│       ├── 05_numerical_vs_target.png
│       ├── 06_correlation_matrix.png
│       ├── 07_model_comparison_cv.png
│       ├── 08_overfitting_analysis.png
│       ├── 09_final_model_evaluation.png
│       └── 10_feature_importance.png
│
├── src/
│   ├── data/                   <- Scripts de carga y limpieza
│   ├── features/               <- Feature engineering
│   └── models/                 <- Entrenamiento y evaluación
│
├── .gitignore                  <- Archivos excluidos de Git
├── README.md                   <- Documentación del proyecto
├── requirements.txt            <- Dependencias del proyecto
```
---

## ⚙️ Instalación y uso

### 1. Clonar el repositorio

```bash
git clone https://github.com/MarialsaDurango/datascope-bank-marketing.git
cd datascope-bank-marketing
```

### 2. Crear entorno virtual

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Descargar el dataset

Descarga `bank-additional-full.csv` desde el
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing)
y colócalo en `data/raw/`.

### 5. Ejecutar el notebook

```bash
jupyter notebook notebooks/bank_marketing_ml_project.ipynb
```

---

## 🔬 Metodología CRISP-DM

| Fase | Estado | Descripción |
|---|---|---|
| ✅ Comprensión del negocio | Completada | Definición del problema y objetivos |
| ✅ Comprensión de los datos | Completada | EDA univariante, bivariante y correlaciones |
| ✅ Preparación de los datos | Completada | Feature engineering, Pipeline, anti-leakage |
| ✅ Modelado | Completada | 4 modelos, Cross-Validation estratificada |
| ✅ Evaluación | Completada | Evaluación en test, matriz de confusión, ROC |
| ✅ Conclusiones | Completada | Interpretación, limitaciones y recomendaciones |

---

## ⚠️ Decisiones técnicas clave

### Exclusión de la variable `duration`
La variable `duration` (duración de la llamada) fue excluida del modelo porque
**no está disponible antes de realizar la llamada** — incluirla sería data leakage.
*(Referencia: Moro et al., 2014)*

### Tratamiento de valores `unknown`
Los valores desconocidos están codificados como `"unknown"` (no como NaN).
Se tratan como **categoría válida adicional** — la ausencia de información
es en sí misma información relevante para el modelo.

### Feature Engineering aplicado
- `was_previously_contacted`: variable binaria derivada de `pdays=999`
- `pdays_clean`: reemplaza el código 999 por 0
- `campaign_capped`: winsorización al percentil 99 para reducir outliers

### Garantía anti-leakage
El Pipeline de scikit-learn garantiza que el preprocesamiento
(StandardScaler, OrdinalEncoder, OneHotEncoder) se ajusta
**únicamente sobre los datos de entrenamiento**.

---

## 📚 Aprendizajes

- La importancia del **contexto macroeconómico** como predictor supera
  al perfil individual del cliente en productos bancarios de ahorro
- El **desbalanceo de clases** hace que el Accuracy sea una métrica
  completamente engañosa — siempre usar F1, Recall y ROC-AUC
- Los **Pipelines de scikit-learn** son la única forma correcta de
  garantizar la ausencia de data leakage en producción
- El **Feature Engineering** (crear `was_previously_contacted` desde
  `pdays=999`) puede ser más valioso que cualquier algoritmo sofisticado
- La **Cross-Validation estratificada** es imprescindible con clases
  desbalanceadas — un KFold normal puede dejar folds sin casos positivos

---

## 🔮 Trabajo futuro

- [ ] Optimización de hiperparámetros con Optuna o GridSearchCV
- [ ] Implementar SMOTE para manejo avanzado del desbalanceo
- [ ] Explorar XGBoost, LightGBM y CatBoost
- [ ] Añadir SHAP values para explicabilidad por instancia
- [ ] Ajustar el umbral de clasificación para maximizar Recall
- [ ] Desplegar el modelo como API REST con FastAPI
- [ ] Implementar monitoreo de drift del modelo en producción
- [ ] Re-entrenar con datos más recientes del mercado europeo

---

## 👩‍💻 Autora

**María Isabel Durango**
Estudiante de Ciencia de Datos
Proyecto desarrollado para DataScope Solutions


---

<div align="center">

*Proyecto desarrollado como parte del portfolio profesional de Data Science*
*Metodología CRISP-DM · Python · Scikit-learn · DataScope Solutions*

**ROC-AUC: 0.811 | Recall: 62.5% | Reducción de llamadas: ~84%**

</div>