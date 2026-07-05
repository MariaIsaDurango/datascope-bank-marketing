<div align="center">

# 🏦 Predicción de Suscripción a Depósitos a Plazo
### DataScope Solutions — Proyecto de Machine Learning

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?style=flat&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-green?style=flat&logo=pandas)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat&logo=jupyter)
![Estado](https://img.shields.io/badge/Estado-En%20desarrollo-yellow?style=flat)
![Metodología](https://img.shields.io/badge/Metodología-CRISP--DM-blue?style=flat)

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
tasa de conversión — todo sin necesidad de contactar a clientes que rechazarán 
la oferta.

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

## 🛠️ Tecnologías utilizadas

| Categoría | Herramientas |
|---|---|
| **Lenguaje** | Python 3.9+ |
| **Manipulación de datos** | Pandas, NumPy |
| **Visualización** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn |
| **Entorno** | Jupyter Notebook |
| **Control de versiones** | Git, GitHub |
| **Metodología** | CRISP-DM |

---

## 📁 Estructura del proyecto