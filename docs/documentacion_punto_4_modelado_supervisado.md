# Documentación del Punto 4: Modelado Supervisado

## 1. Objetivo

El objetivo de esta etapa fue entrenar y evaluar modelos de aprendizaje supervisado
para clasificar la severidad de los siniestros de tránsito en el Perú.

El notebook 04 toma como entrada los archivos generados por el notebook 03:

`data/procesada/siniestros_train.csv`

`data/procesada/siniestros_test.csv`

`data/procesada/siniestros_modelado.csv`

El archivo de entrenamiento contiene datos balanceados con SMOTE, mientras que el archivo
de prueba conserva la distribución real de los siniestros. Esto permite entrenar modelos
menos sesgados hacia la clase mayoritaria y evaluarlos de forma realista.

## 2. Motivo del Modelado Supervisado

Después de preparar los datos en el notebook 03, era necesario construir modelos capaces
de aprender patrones asociados a la severidad del siniestro.

El problema se trata como una tarea de clasificación multiclase, porque la variable objetivo
puede tomar tres valores:

- `GRAVE_LESIONADOS`
- `GRAVE_MORTALIDAD`
- `LEVE`

No se eligió evaluar únicamente con accuracy, porque la clase LEVE representa la mayoría
de los registros. Si el modelo predijera casi siempre LEVE, podría obtener un accuracy alto,
pero fallaría en los casos más importantes para el estudio: los siniestros graves.

Por ese motivo, se priorizaron métricas como `f1_macro` y `recall_macro`, que evalúan el
rendimiento del modelo considerando todas las clases de forma más equilibrada.

## 3. Pasos Realizados En El Notebook 04

Se creó el notebook:

`notebooks/04_modelado_supervisado.ipynb`

El notebook realiza los siguientes pasos:

1. Carga los datasets generados por el notebook 03.
2. Valida que train y test contengan la variable objetivo y no tengan valores nulos.
3. Separa las variables predictoras (`X`) y la variable objetivo (`y`).
4. Entrena un modelo Random Forest.
5. Entrena un modelo XGBoost.
6. Evalúa ambos modelos con métricas de clasificación multiclase.
7. Genera matrices de confusión para analizar aciertos y errores.
8. Calcula la importancia de variables de cada modelo.
9. Aplica SHAP para interpretar globalmente el mejor modelo.
10. Exporta métricas, tablas y figuras para el informe final.

## 4. Validación de Datos

El notebook cargó inicialmente los siguientes archivos:

- Train: 18,363 filas x 46 columnas.
- Test: 3,263 filas x 46 columnas.
- Dataset completo sin SMOTE: 9,106 filas x 46 columnas.

Durante la validación se detectó que el archivo `siniestros_test.csv` tenía 1,441 filas con
valores nulos en la variable objetivo y en las features. Esto ocurrió por una desalineación
de índices al exportar el test desde el notebook 03.

Para evitar que el modelo se entrenara o evaluara con datos incorrectos, el notebook 04
reconstruyó el conjunto de prueba desde `siniestros_modelado.csv`, usando un split
estratificado 80/20 con `random_state=42`.

Resultado usado para el modelado:

- `X_train`: 18,363 registros x 44 features.
- `X_test`: 1,822 registros x 44 features.

Distribución del entrenamiento:

- `GRAVE_LESIONADOS`: 6,121 registros.
- `GRAVE_MORTALIDAD`: 6,121 registros.
- `LEVE`: 6,121 registros.

Distribución del test:

- `GRAVE_LESIONADOS`: 105 registros.
- `GRAVE_MORTALIDAD`: 186 registros.
- `LEVE`: 1,531 registros.

Esto confirma que el entrenamiento quedó balanceado por SMOTE y que la evaluación se hizo
sobre datos reales no balanceados.

## 5. Modelos Entrenados

Se entrenaron dos modelos supervisados:

**Random Forest**

Random Forest construye muchos árboles de decisión y combina sus resultados. Es útil porque
trabaja bien con variables numéricas y categóricas codificadas, maneja relaciones no lineales
y permite obtener importancia de variables.

**XGBoost**

XGBoost también usa árboles de decisión, pero los entrena de forma secuencial. Cada nuevo
árbol intenta corregir errores cometidos por los árboles anteriores. Por eso suele obtener
buen rendimiento en problemas tabulares de clasificación.

Ambos modelos fueron entrenados con el conjunto balanceado por SMOTE y evaluados con el
conjunto de prueba real.

## 6. Métricas de Evaluación

Las métricas calculadas fueron:

- `accuracy`: porcentaje total de predicciones correctas.
- `precision_macro`: precisión promedio considerando todas las clases por igual.
- `recall_macro`: capacidad promedio de encontrar correctamente cada clase.
- `f1_macro`: equilibrio entre precisión y recall para todas las clases.
- `precision_weighted`, `recall_weighted` y `f1_weighted`: métricas ponderadas por cantidad de registros.
- `roc_auc_ovr_macro`: capacidad del modelo para separar las clases en un enfoque One-vs-Rest.

Se priorizó `f1_macro` porque el problema tiene clases desbalanceadas y las clases graves
son minoritarias.

## 7. Resultados Obtenidos

Resultados principales:

| Modelo | Accuracy | Precision Macro | Recall Macro | F1 Macro | ROC-AUC OVR Macro |
|---|---:|---:|---:|---:|---:|
| XGBoost | 0.7942 | 0.4231 | 0.4048 | 0.4104 | 0.7605 |
| Random Forest | 0.7870 | 0.4048 | 0.4026 | 0.4015 | 0.7482 |

El mejor modelo según `f1_macro` fue:

`XGBoost`

XGBoost superó ligeramente a Random Forest en F1 macro, recall macro y ROC-AUC macro.

Sin embargo, los resultados también muestran que el modelo identifica mucho mejor la clase
LEVE que las clases graves. Esto es esperable porque en el conjunto real de prueba la clase
LEVE sigue siendo dominante.

Resultados por clase para XGBoost:

| Clase | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| GRAVE_LESIONADOS | 0.1887 | 0.0952 | 0.1266 | 105 |
| GRAVE_MORTALIDAD | 0.2606 | 0.2312 | 0.2450 | 186 |
| LEVE | 0.8703 | 0.9118 | 0.8906 | 1,531 |

La clase `GRAVE_LESIONADOS` tiene el F1 más bajo. Esto significa que el modelo detecta pocos
casos de esta clase y, cuando la predice, todavía se equivoca con frecuencia.

## 8. Matrices de Confusión

Se generaron matrices de confusión para Random Forest y XGBoost.

Estas matrices permiten observar cuántos casos fueron clasificados correctamente y cuántos
fueron confundidos con otra clase.

Resultado general observado:

- Ambos modelos predicen correctamente una gran cantidad de casos LEVE.
- Las clases `GRAVE_LESIONADOS` y `GRAVE_MORTALIDAD` son más difíciles de detectar.
- XGBoost mejora ligeramente la detección de clases graves frente a Random Forest.

Esto indica que el modelo sí aprende patrones útiles, pero todavía tiene limitaciones para
distinguir casos graves debido a la baja cantidad de ejemplos reales en el test.

## 9. Interpretabilidad con Importancia de Variables

El notebook calcula la importancia de variables para Random Forest y XGBoost.

En XGBoost, las variables más importantes fueron:

- `ZONA`
- `CLASE SINIESTRO`
- `PERFIL LONGITUDINAL VÍA`
- `MTC_RVN_SUPERFICIE_DOMINANTE`
- `CARACTERÍSTICAS DE VÍA`
- `ES_FIN_SEMANA`
- `MTC_RVN_CONCESION_PRINCIPAL`
- `ZONIFICACIÓN`
- `MTC_RVN_ESTADO_DOMINANTE`
- `RED VIAL`

Estas variables sugieren que el modelo utiliza información del tipo de siniestro, la zona,
las características de la vía, condiciones de la red vial y factores temporales para estimar
la severidad.

## 10. Interpretabilidad con SHAP

Además de la importancia interna de los modelos, se aplicó SHAP sobre XGBoost para explicar
el impacto global de las variables.

SHAP ayuda a responder una pregunta sencilla:

`¿Qué variables empujan más las predicciones del modelo hacia una u otra clase?`

Las variables con mayor importancia SHAP fueron:

- `CLASE SINIESTRO`
- `ZONA`
- `HORA_NUM`
- `FRANJA_HORARIA`
- `CARACTERÍSTICAS DE VÍA`
- `IDH_POBLACION_2019`
- `MES`
- `AÑO`
- `MTC_RVN_SUPERFICIE_DOMINANTE`
- `ES_FIN_SEMANA`
- `¿EXISTE SEÑAL VERTICAL?`
- `CAUSA FACTOR PRINCIPAL`
- `COORDENADAS LONGITUD`
- `PERFIL LONGITUDINAL VÍA`
- `RED VIAL`

Esto permite justificar que el modelo no funciona como una caja negra, sino que sus decisiones
pueden ser interpretadas a partir de variables relacionadas con ubicación, tiempo, vía,
señalización, causa del accidente e indicadores externos.

Además, se agregó un beeswarm SHAP por clase usando el mejor modelo (`XGBoost`). Este gráfico
muestra la dirección y la intensidad del impacto de cada variable en cada clase. La lectura
principal es que para `GRAVE_LESIONADOS` no aparece una variable con valores SHAP
consistentemente altos que empuje de forma clara las predicciones hacia esa clase. Esto ayuda
a explicar por qué su F1-score es bajo: el modelo no encuentra una señal fuerte y estable
para diferenciar los siniestros con lesionados graves del resto.

Mantener separadas las clases `GRAVE_LESIONADOS` y `GRAVE_MORTALIDAD` sigue siendo adecuado
desde la perspectiva de seguridad vial. Los siniestros con lesionados no fatales y los
siniestros con víctimas fatales pueden responder a factores, costos e intervenciones distintas,
por lo que tratarlos como categorías separadas conserva información relevante para el análisis.

## 11. Archivos Generados

El notebook genera automáticamente los siguientes archivos:

`outputs/tablas/metricas_modelos.csv`

Tabla comparativa con las métricas principales de Random Forest y XGBoost.

`outputs/tablas/metricas_por_clase.csv`

Tabla con precision, recall, F1-score y support para cada clase y cada modelo.

`outputs/tablas/importancias_random_forest.csv`

Tabla con la importancia de variables calculada por Random Forest.

`outputs/tablas/importancias_xgboost.csv`

Tabla con la importancia de variables calculada por XGBoost.

`outputs/tablas/importancia_shap.csv`

Tabla con la importancia global calculada mediante SHAP.

`outputs/figures/confusion_random_forest.png`

Matriz de confusión del modelo Random Forest.

`outputs/figures/confusion_xgboost.png`

Matriz de confusión del modelo XGBoost.

`outputs/figures/comparacion_modelos.png`

Gráfico comparativo de accuracy, F1 macro y recall macro.

`outputs/figures/importancias_random_forest.png`

Gráfico de las principales variables usadas por Random Forest.

`outputs/figures/importancias_xgboost.png`

Gráfico de las principales variables usadas por XGBoost.

`outputs/figures/shap_importancia_global.png`

Gráfico global de importancia SHAP.

`outputs/figures/shap_summary.png`

Gráfico resumen SHAP que muestra impacto y dirección de las variables.

`outputs/figures/shap_beeswarm_GRAVE_LESIONADOS.png`

Beeswarm SHAP para la clase GRAVE_LESIONADOS.

`outputs/figures/shap_beeswarm_GRAVE_MORTALIDAD.png`

Beeswarm SHAP para la clase GRAVE_MORTALIDAD.

`outputs/figures/shap_beeswarm_LEVE.png`

Beeswarm SHAP para la clase LEVE.

## 12. Resumen Final

Dataset de entrenamiento: 18,363 registros balanceados con SMOTE.

Dataset de prueba: 1,822 registros reales sin SMOTE.

Features utilizadas: 44.

Modelos entrenados: Random Forest y XGBoost.

Mejor modelo según F1 macro: XGBoost.

F1 macro de XGBoost: 0.4104.

ROC-AUC OVR macro de XGBoost: 0.7605.

Variables más relevantes según SHAP: clase de siniestro, zona, hora, franja horaria,
características de vía, IDH, superficie dominante, señalización, causa principal y red vial.

Conclusión metodológica: el modelo XGBoost fue el mejor de los modelos probados, aunque el
rendimiento en las clases graves todavía es limitado. Esto no invalida el modelo; al contrario,
muestra una dificultad real del problema: los casos graves son menos frecuentes y más difíciles
de distinguir usando solo las variables disponibles.

## 13. Referencias

Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system.
Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.

Géron, A. (2022). Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow
(3rd ed.). O'Reilly Media.

Han, J., Kamber, M., & Pei, J. (2012). Data Mining: Concepts and Techniques.
Morgan Kaufmann.

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions.
Advances in Neural Information Processing Systems, 30.
