# Siniestros Fatales Perú — Modelado Predictivo y Análisis Espacial

Análisis predictivo de la severidad de siniestros fatales de tránsito en el Perú (2021–2025) mediante aprendizaje automático explicable y detección de zonas de riesgo.

**Curso:** Agentes Inteligentes — FC-VIRISF08A01  
**Institución:** Facultad de Ingeniería, Ingeniería de Sistemas de Información, Software y Ciencia de Datos  
**Grupo 5**

---

## Estructura del repositorio

```
siniestros-fatales-peru-ml/
│
├── data/
│   ├── raw/              # Dataset original descargado del ONSV (.xlsx)
│   ├── procesada/        # Datasets generados por los notebooks
│   └── externas/         # Fuentes externas MTC, IDH, SENAMHI (no versionadas)
│
├── notebooks/
│   ├── 01_EDA_limpieza.ipynb
│   ├── 02_enriquecimiento_externo.ipynb
│   ├── 03_preparacion_modelado.ipynb
│   ├── 04_modelado_supervisado.ipynb       
│   └── 05_hotspots_dbscan.ipynb            
│
├── outputs/
│   └── figures/          # Gráficos generados por los notebooks
│
├── README.md
└── requirements.txt
```

---

## Pipeline metodológico

El flujo se organizó en cinco fases derivadas de Han, Kamber y Pei (2012), que distinguen *data cleaning* y *data transformation* como etapas separadas del preprocesamiento, y de Géron (2022), que define el pipeline general de ML como marco ordenador.

| Notebook | Fase | Estado | Referencia |
|---|---|---|---|
| 01 — EDA y Limpieza | Exploración + *data cleaning* |  Completo | Han et al. (2012) |
| 02 — Enriquecimiento externo | Integración de fuentes externas (MTC, IDH/PNUD, SINAC) |  Completo | Provost & Fawcett (2013) |
| 03 — Preparación para modelado | *Data transformation* + selección de features + SMOTE |  Completo | Géron (2022) |
| 04 — Modelado supervisado | Random Forest + XGBoost + SHAP |  Pendiente | Breiman (2001); Chen et al. (2025) |
| 05 — Hotspots DBSCAN | Clustering espacial de zonas de riesgo |  Pendiente | Kamh et al. (s.f.) |

---

## Dataset

**Fuente:** Observatorio Nacional de Seguridad Vial — [Datos Abiertos](https://www.onsv.gob.pe/datosabiertos)  
**Periodo:** 2021–2025 (2024–2025 preliminar)  
**Registros:** 9,106 siniestros fatales  
**Variables originales:** 27  
**Variables tras limpieza y feature engineering:** 36

El archivo raw no se sube al repositorio por su tamaño. Descargarlo directamente desde el portal del ONSV y colocarlo en `data/raw/`.

---

## Requisitos

Dependencias principales: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `imbalanced-learn`, `openpyxl`.

---

## Referencias

- Bazarnovi & Mohammadian (2024) — SMOTE en datasets de accidentes de tránsito
- Breiman (2001) — Random Forests
- Chen et al. (2025) — XGBoost + SHAP en severidad vial
- Géron (2022) — *Hands-On Machine Learning*, O'Reilly
- Han, Kamber & Pei (2012) — *Data Mining: Concepts and Techniques*, Morgan Kaufmann
- Mokoatle et al. (2019) — Variable objetivo multicategórica de severidad
- Provost & Fawcett (2013) — *Data Science for Business*, O'Reilly