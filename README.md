# Siniestros Fatales en el Perú: Modelado Predictivo y Análisis Espacial (2021–2025)

Facultad de Ingeniería - Ingeniería de Sistemas de Información, Software y Ciencia de Datos
Universidad San Ignacio de Loyola | Curso: Agentes Inteligentes | 2026-01

---

## Sobre el proyecto

Este proyecto analiza los factores asociados a la severidad de siniestros fatales de tránsito en el Perú entre 2021 y 2025. La pregunta central no es si un accidente fue fatal, sino qué factores determinan que resulte en víctimas múltiples y dónde se concentran geográficamente esos eventos.

Se usa el dataset público del Observatorio Nacional de Seguridad Vial (ONSV), enriquecido con fuentes del Ministerio de Transportes y Comunicaciones, el IDH distrital del PNUD, y los datasets relacionales de personas y vehículos del propio ONSV. Sobre esa base se aplican modelos de clasificación supervisada con interpretabilidad SHAP y clustering espacial DBSCAN.

---

## Equipo

| Integrante | Notebook |
|---|---|
| Bautista Arrilucea, Romina Zoraida Perla | NB02 Enriquecimiento |
| Riva Ramos, Renato Sebastian | NB04 Modelado supervisado |
| Buitrón Catalan, Freddy Armando | NB03 Preparación para modelado |
| Veliz Garcia, Alejandra | NB01 EDA y Limpieza / NB05 DBSCAN / Coordinación |

---

## Estructura del repositorio

```
siniestros-fatales-peru-ml/
│
├── data/
│   ├── raw/              Dataset original ONSV (.xlsx) — no versionado por tamaño
│   ├── procesada/        Datasets intermedios y finales generados por el pipeline
│   └── externas/         Fuentes MTC, IDH y ONSV relacionales — no versionadas
│
├── notebooks/
│   ├── 01_EDA_limpieza.ipynb
│   ├── 02_enriquecimiento_externo.ipynb
│   ├── 03_preparacion_modelado.ipynb
│   ├── 04_modelado_supervisado.ipynb
│   └── 05_hotspots_dbscan.ipynb
│
├── outputs/
│   ├── figures/          Gráficos generados por los notebooks
│   └── tablas/           Métricas y resúmenes en CSV
│
├── README.md
└── requirements.txt
```

El dataset raw del ONSV no se versiona por tamaño. Descargarlo desde el portal de datos abiertos del ONSV (https://www.onsv.gob.pe/datosabiertos) y colocarlo en `data/raw/`.

---

## Pipeline

Cada notebook produce un output verificado que es la entrada del siguiente. Ninguna decisión de modelado se toma sobre datos sin limpiar, y el enriquecimiento externo ocurre antes de la selección de features para preservar las llaves de cruce.

| Notebook | Fase | Output |
|---|---|---|
| 01 | EDA y limpieza | siniestros_limpio.csv |
| 02 | Enriquecimiento MTC, IDH y ONSV personas/vehículos | siniestros_enriquecido.csv |
| 03 | Feature engineering, encoding y SMOTE | siniestros_train/test/modelado.csv |
| 04 | Random Forest, XGBoost y SHAP | Métricas y gráficos |
| 05 | Clustering espacial DBSCAN | siniestros_clusters.csv |

---

## Resultados principales

XGBoost obtuvo el mejor desempeño con F1 macro de 0.501 y accuracy de 0.815, evaluado sobre la distribución real sin balanceo. La diferencia frente a Random Forest es de 0.002 puntos, lo que no representa superioridad estadística fuerte.

El análisis SHAP identificó que la clase de siniestro, la presencia de motocicletas, la presencia de camiones o buses, la zona y la presencia de peatones son los factores con mayor peso sobre la predicción de severidad.

El clustering DBSCAN detectó 51 zonas de concentración. Áncash registró la mayor letalidad relativa con 14% de casos de mortalidad grave, mientras que Lima Metropolitana presentó la más baja con 4.3%, atribuible a la densidad de servicios de emergencia.

---

## Instalación y uso

Clonar el repositorio e instalar las dependencias:

```bash
git clone https://github.com/veliz-a/siniestros-fatales-peru-ml
cd siniestros-fatales-peru-ml
pip install -r requirements.txt
```

Los notebooks deben ejecutarse en orden del 01 al 05. Cada uno detecta automáticamente si se ejecuta desde la raíz del repositorio o desde la carpeta `notebooks/`.

Las dependencias principales son las siguientes. Se recomienda Python 3.10 o superior.

```
pandas>=2.0
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
shap
openpyxl
```

---

## Referencias

Bazarnovi, S. y Mohammadian, A. K. (2024). Addressing imbalanced data in predicting injury severity after traffic crashes. Procedia Computer Science, 238, 24-31.

Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.

Chen, F. et al. (2025). Traffic accident severity prediction based on an enhanced MSCPO-XGBoost hybrid model. Scientific Reports, 15(1), 797.

Ester, M. et al. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. KDD-96 Proceedings, 226-231.

Gao, X. et al. (2025). Reliable imputation of incomplete crash data for predicting driver injury severity. Accident Analysis and Prevention, 216, 108020.

Géron, A. (2022). Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow (3rd ed.). O'Reilly Media.

Han, J., Kamber, M. y Pei, J. (2012). Data Mining: Concepts and Techniques. Morgan Kaufmann.

Kamh, H. et al. (s.f.). Exploring Road Traffic Accidents Hotspots Using Clustering Algorithms and GIS-based Spatial Analysis. IEEE Access.

McKinney, W. (2022). Python for Data Analysis (3rd ed.). O'Reilly Media.

Mokoatle, M., Marivate, V. y Esiefarienrhe, M. B. (2019). Predicting Road Traffic Accident Severity using Accident Report Data in South Africa. dg.o 2019, 1-9.

Observatorio Nacional de Seguridad Vial. (2026). Datos abiertos. https://www.onsv.gob.pe/datosabiertos
