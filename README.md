# siniestros-fatales-peru-ml
# Análisis Predictivo y Detección de Zonas de Riesgo en Siniestros Fatales de Tránsito en el Perú (2021–2025) mediante Aprendizaje Automático Explicable
**Grupo 5 | Agentes Inteligentes**

---
## 1. Estructura del pipeline metodológico
 
El flujo de trabajo se organizó en cinco fases derivadas de dos referencias centrales. Han, Kamber y Pei (2012) distinguen explícitamente entre *data cleaning* y *data transformation* como fases separadas del preprocesamiento, lo que justifica tener notebooks independientes para cada etapa. Géron (2022) define el pipeline general de ML como: obtención de datos → exploración → preparación → modelado → ajuste, que sirve como marco ordenador del conjunto.
 
| Notebook | Fase | Referencia principal |
|---|---|---|
| 01 — EDA y Limpieza | Exploración + *data cleaning* | Han et al. (2012) |
| 02 — Enriquecimiento externo | Integración de fuentes externas | Provost & Fawcett (2013) |
| 03 — Preparación para modelado | *Data transformation* + selección de features + SMOTE | Géron (2022) |
| 04 — Modelado supervisado | Random Forest + XGBoost + SHAP | Breiman (2001); Chen et al. (2025) |
| 05 — Hotspots DBSCAN | Clustering espacial | Kamh et al. (s.f.) |
