# ML_Project_Loan_Status

El link del repo de github es el siguiente: https://github.com/ramonmorillx/ML_Project_Loan_Status

**Autor**:
Ramón Morillo Barrera - ramon.morillo@cunef.edu / ramonvejer@gmail.com

## Bienvenidos a mi repo de Github

En este repositorio registro todo lo referente a mi proyecto más grande de ML hasta la fecha, se trata de un proyecto completo de ML que incluye un EDA, feature processing, model selection, model implementation, explainability y conclusions. Este repositorio se irá actualizando conforme vaya avanzando con el proyecto, ya que es parte del Máster en ciencia de datos que estoy cursando.

## Machine Learning Project:

### Resumen del Proyecto:

Este proyecto tiene como objetivo aplicar los conocimientos adquiridos en la asignatura de Machine Learning para desarrollar un modelo capaz de detectar clientes con dificultad de pago a la hora de conceder u préstamo bancario. Se realizará un análisis exploratorio de los datos, seguido de un preprocesamiento adecuado, y se abordará el desbalanceo de la variable objetivo. Además, se aplicarán diversos modelos de Machine Learning, se explicará su funcionamiento y se presentarán los resultados obtenidos.

### Objetivos Principales:

**Objetivos Generales**:

- Aplicar los conocimientos adquiridos en la asignatura de Machine Learning en un proyecto práctico y real.
- Profundizar en el análisis y procesamiento de datos, así como en los diferentes modelos de clasificación que se utilizarán.
- Comprender el proceso de modelado en un proyecto real, adoptando un enfoque de aprendizaje autónomo y mediante un proceso de prueba y error.

**Análisis Exploratorio de Datos (EDA)**:

- Familiarizarse con el conjunto de datos que se utilizará.
- Analizar las variables que lo componen y realizar una revisión general de las instancias.
- Entender la distribución de los datos, las correlaciones entre las variables y especialmente con la variable objetivo (target).
- Extraer conclusiones sobre la estructura del conjunto de datos, sus componentes principales y los hallazgos derivados del análisis exploratorio.

**Aplicación de Modelos Predictivos**:

- Realizar correctamente el procesamiento de las variables y su selección para aplicar los modelos de manera efectiva.
- Comparar diferentes modelos de Machine Learning y comprender las razones por las que algunos funcionan mejor que otros.
- Establecer los métodos más adecuados para evaluar el rendimiento de los modelos en cada caso.
- Abordar el problema del desbalanceo de clases, explorar técnicas que ayuden a manejarlo y verificar su efectividad en este contexto específico.
- Evaluar el impacto de distintas configuraciones de datos sobre las métricas seleccionadas.
- Mejorar el rendimiento de los modelos mediante técnicas como la validación cruzada (Cross Validation) y la optimización de hiperparámetros (Hyperparameter Tuning).
- Aplicar los modelos seleccionados, medir su desempeño y analizar los resultados obtenidos.

**Explicabilidad**:

- En esta sección, se ofrecerán explicaciones sobre el funcionamiento interno de los modelos, utilizando herramientas específicas para evaluar su interpretabilidad.
- El objetivo será entender cómo operan los modelos, identificar qué variables son más influyentes en las decisiones del modelo, y analizar cómo varían las decisiones del modelo dependiendo de los distintos casos evaluados, entre otros aspectos clave.

#### Repositorio de Github:
El repositorio está compuesto por las siguientes carpetas:

- **`data`**: Contiene los archivos de datos con los que vamos a trabajar.

  - **`raw`**: Archivos de datos originales, tal como se obtuvieron.
 
  - **`processed`**: Datos que ya han sido procesados y transformados para su uso.
  
  - **`interim`**: Datos intermedios que han sido parcialmente procesados y aún no están listos para su uso final.
  
- **`src`**: directorio que contiene las funciones .py auxiliares que se importan en los notebooks

- **`html`**: carpeta en donde se encuentran todos los archivos html exportados

- **`notebooks`**: carpeta que contiene los notebooks con todos los procesos realizados

- **`experiments`**: carpeta que alberga diferentes experimentos realizados que justifican algunas decisiones tomadas durante el proceso.

- **`env`**: carpeta con los requerimientos del enviroment para poder ejecutar todo el código sin problemas.

Los datos debido a su capacidad de almacenamiento no estarán en github, pero se pueden solicitar bajo petición anticipada.

## Notebooks Desarrollados

Se han desarrollado tres notebooks en esta primera entrega:

1. **01_Problem_EDA.ipynb**: Dedicado a analizar y comprender las variables del conjunto de datos, así como a examinar su distribución y características principales.
2. **02_EDA_Final.ipynb**: Orientado a realizar la división del conjunto de datos en subconjuntos de entrenamiento y prueba, identificar valores nulos y atípicos, y explorar las correlaciones entre las diferentes variables.
3. **03_Feature_Engineering.ipynb**: Enfocado en transformar las variables categóricas mediante técnicas de codificación y normalizar las variables numéricas a través de métodos de escalado, además de la selección de las mismas.
4. **04_Model_Selection.ipynb**: Dirigido a la implmentación de modelos de predicción y análisis de las diferentes métricas observadas.
5. **05_Final_Model.ipynb**: Dedicado a la implentación del modelo seleccionado, evaluación de sus principales curvas,  métricas y comparación de su efectividad con diferentes tresholds.
6. **06_Explainability.ipynb**: Orientado a la explicabilidad del modelo con la librería SHAP, donde se incide tanto en explicabilidad global como en explicabilidad local.
7. **07_Conclusion.ipynb**: Recapitulación de todo el proceso llevado a cabo en el proyecto, resaltando los hallazgos y características más importantes del mismo.