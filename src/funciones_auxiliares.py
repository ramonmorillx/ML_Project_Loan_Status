import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
from termcolor import colored, cprint # type: ignore
import warnings
import importlib
import scipy.stats as ss
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import TargetEncoder
from category_encoders import TargetEncoder, CatBoostEncoder

from sklearn.metrics import balanced_accuracy_score, classification_report, roc_auc_score, fbeta_score, make_scorer,\
                            accuracy_score,average_precision_score, precision_recall_curve, roc_curve,\
                            auc, recall_score, precision_score, confusion_matrix, f1_score, ConfusionMatrixDisplay
                            
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
from sklearn.svm import SVC, LinearSVC, NuSVC
from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

seed=12354

###-------------------------------- Funciones del máster utilizadas ---------------------------------------
def duplicate_columns(frame):
    '''
    Lo que hace la función es, en forma de bucle, ir seleccionando columna por columna del DF que se le indique
    y comparar sus values con los de todas las demás columnas del DF. Si son exactamente iguales, añade dicha
    columna a una lista, para finalmente devolver la lista con los nombres de las columnas duplicadas.
    '''
    groups = frame.columns.to_series().groupby(frame.dtypes).groups
    dups = []

    for t, v in groups.items():

        cs = frame[v].columns
        vs = frame[v]
        lcs = len(cs)

        for i in range(lcs):
            ia = vs.iloc[:,i].values
            for j in range(i+1, lcs):
                ja = vs.iloc[:,j].values
                if np.array_equal(ia, ja):
                    dups.append(cs[i])
                    break
    return dups

### -----------------------

def dame_variables_categoricas(dataset=None):
    '''
    ----------------------------------------------------------------------------------------------------------
    Función dame_variables_categoricas:
    ----------------------------------------------------------------------------------------------------------
        -Descripción: Función que recibe un dataset y devuelve una lista con los nombres de las 
        variables categóricas
        -Inputs: 
            -- dataset: Pandas dataframe que contiene los datos
        -Return:
            -- lista_variables_categoricas: lista con los nombres de las variables categóricas del
            dataset de entrada con menos de 100 valores diferentes
            -- 1: la ejecución es incorrecta
    '''
    if dataset is None:
        print(u'\nFaltan argumentos por pasar a la función')
        return 1
    lista_variables_categoricas = []
    other = []
    for i in dataset.columns:
        if (dataset[i].dtype!=float) & (dataset[i].dtype!=int):
            unicos = int(len(np.unique(dataset[i].dropna(axis=0, how='all'))))
            if unicos < 100:
                lista_variables_categoricas.append(i)
            else:
                other.append(i)

    return lista_variables_categoricas, other

### ----

def get_deviation_of_mean_perc(pd_loan, list_var_continuous, target, multiplier):
    """
    Devuelve el porcentaje de valores que exceden del intervalo de confianza
    :type series:
    :param multiplier:
    :return:
    """
    pd_final = pd.DataFrame()
    
    for i in list_var_continuous:
        
        series_mean = pd_loan[i].mean()
        series_std = pd_loan[i].std()
        std_amp = multiplier * series_std
        left = series_mean - std_amp
        right = series_mean + std_amp
        size_s = pd_loan[i].size
        
        perc_goods = pd_loan[i][(pd_loan[i] >= left) & (pd_loan[i] <= right)].size/size_s
        perc_excess = pd_loan[i][(pd_loan[i] < left) | (pd_loan[i] > right)].size/size_s
        
        if perc_excess>0:    
            pd_concat_percent = pd.DataFrame(pd_loan[target][(pd_loan[i] < left) | (pd_loan[i] > right)]\
                                            .value_counts(normalize=True).reset_index()).T
            pd_concat_percent.columns = [pd_concat_percent.iloc[0,0], 
                                         pd_concat_percent.iloc[0,1]]
            pd_concat_percent = pd_concat_percent.drop(target,axis=0)
            pd_concat_percent['variable'] = i
            pd_concat_percent['sum_outlier_values'] = pd_loan[i][(pd_loan[i] < left) | (pd_loan[i] > right)].size
            pd_concat_percent['porcentaje_sum_null_values'] = perc_excess
            pd_final = pd.concat([pd_final, pd_concat_percent], axis=0).reset_index(drop=True)
            
    if pd_final.empty:
        print('No existen variables con valores nulos')
        
    return pd_final

#####

def cramers_v(confusion_matrix):
    """ 
    calculate Cramers V statistic for categorial-categorial association.
    uses correction from Bergsma and Wicher,
    Journal of the Korean Statistical Society 42 (2013): 323-328
    
    confusion_matrix: tabla creada con pd.crosstab()
    
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))
    
    

#####

def get_percent_null_values_target(pd_loan, list_var_continuous, target):

    pd_final = pd.DataFrame()
    for i in list_var_continuous:
        if pd_loan[i].isnull().sum()>0:
            pd_concat_percent = pd.DataFrame(pd_loan[target][pd_loan[i].isnull()]\
                                            .value_counts(normalize=True).reset_index()).T
            pd_concat_percent.columns = [pd_concat_percent.iloc[0,0], 
                                         pd_concat_percent.iloc[0,1]]
            pd_concat_percent = pd_concat_percent.drop('index',axis=0)
            pd_concat_percent['variable'] = i
            pd_concat_percent['sum_null_values'] = pd_loan[i].isnull().sum()
            pd_concat_percent['porcentaje_sum_null_values'] = pd_loan[i].isnull().sum()/pd_loan.shape[0]
            pd_final = pd.concat([pd_final, pd_concat_percent], axis=0).reset_index(drop=True)
            
    if pd_final.empty:
        print('No existen variables con valores nulos')
        
    return pd_final


###----------------------------------------- Funciones Propias -----------------------------------------------

def obtener_tipos_valores(df):   
    """
    Esta función toma un DataFrame de pandas como entrada y determina el tipo de 
    cada columna (Categorica, Booleana o Numerica) basado en los valores y el tipo de datos.
    También obtiene los valores únicos de cada columna (excluyendo NaN).
    
    Parámetros:
    ----------
    df : pandas.DataFrame
        El DataFrame cuyas columnas serán analizadas.
    """
    
    tipos = []
    valores = []
    
    for col in df.columns:
        unique_vals = df[col].dropna().unique()
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            tipo = 'Categorica'
        elif df[col].dtype == 'bool' or (set(unique_vals) == {0, 1}):
            tipo = 'Booleana'
        else:
            tipo = 'Numerica'
        
        tipos.append(tipo)
        valores.append(unique_vals)
    
    return pd.DataFrame({'Variable': df.columns, 'Tipo': tipos, 'Valores': valores})

#### 

def lista_valores(df):
    """
    Esta función toma un DataFrame y devuelve tres listas:
    - Variables booleanas
    - Variables categóricas
    - Variables numéricas
    """
    bool_vars = [col for col in df.columns if df[col].dropna().nunique() == 2 and set(df[col].dropna().unique()) == {0, 1}]
    cat_vars = df.select_dtypes(include=['category']).columns.tolist()
    num_vars = df.select_dtypes(include=['number']).columns.tolist()
    
    return bool_vars, cat_vars, num_vars


####


def calculate_woe_iv_categorical(df, variable, target):
    """
    Calcula el WoE y IV de una variable categórica.
    
    Args:
    df (pd.DataFrame): DataFrame con los datos.
    variable (str): Nombre de la variable categórica para la que calcular el WoE y IV.
    target (str): Nombre de la variable objetivo (target).
    
    Returns:
    tuple: Un diccionario con WoE por cada categoría y el valor del IV.
    """
    
    # Inicializamos el diccionario para almacenar los valores de WoE por categoría
    woe_dict = {}
    iv = 0
    
    # Contamos los total de positivos y negativos en el target
    total_positivos = df[df[target] == 1].shape[0]
    total_negativos = df[df[target] == 0].shape[0]
    
    # Agrupamos por la variable categórica y calculamos los conteos de positivos y negativos
    group = df.groupby(variable)[target].value_counts().unstack().fillna(0)
    
    for category in group.index:
        positivos_categoria = group.loc[category, 1]  # positivos en esa categoría
        negativos_categoria = group.loc[category, 0]  # negativos en esa categoría
        
        # Calculamos las proporciones de positivos y negativos en cada categoría
        p_positivos_categoria = positivos_categoria / total_positivos
        p_negativos_categoria = negativos_categoria / total_negativos
        
        # Si alguna proporción es cero, asignamos un WoE de cero (evitar log(0))
        if p_negativos_categoria == 0 or p_positivos_categoria == 0:
            woe = 0
        else:
            woe = np.log(p_positivos_categoria / p_negativos_categoria)
        
        # Almacenamos el WoE del bin
        woe_dict[category] = woe
        
        # Calculamos el IV (Information Value)
        iv += (p_positivos_categoria - p_negativos_categoria) * woe
    
    return woe_dict, iv

######

##### 
def corr_cat_boolean(df):
    """
    Calcula la matriz de correlaciones entre variables booleanas utilizando Cramér's V.
    Convierte columnas con valores int a booleanas antes del cálculo.
    
    Args:
    - df (pd.DataFrame): DataFrame con columnas booleanas o enteras.

    Returns:
    - pd.DataFrame: Matriz de correlaciones basada en Cramer's V.
    """
    # Transformar columnas int a booleanas
    df = df.applymap(lambda x: bool(x) if isinstance(x, (int, np.integer)) else x)
    
    # Verificar que todas las columnas sean booleanas después de la transformación
    if not all(df.dtypes == 'bool'):
        raise ValueError("El DataFrame debe contener únicamente columnas booleanas después de la transformación.")
    
    # Asegurarse de que los datos booleanos estén correctamente codificados como 'True' y 'False'
    df = df.astype(bool)
    
    # Calcular Cramér's V para cada par de columnas
    columns = df.columns
    corr_matrix = []
    for col1 in columns:
        row = []
        for col2 in columns:
            # Crear la matriz de confusión
            confusion_matrix = pd.crosstab(df[col1], df[col2])
            
            # Verificar si la tabla de contingencia tiene más de una categoría
            if confusion_matrix.shape[0] > 1 and confusion_matrix.shape[1] > 1:
                v = cramers_v(confusion_matrix.values)
                row.append(v)
            else:
                # Si no se puede calcular Cramér's V, se asigna un valor de 0
                row.append(0)
        corr_matrix.append(row)

    # Convertir la matriz a un DataFrame
    corr_matrix = pd.DataFrame(corr_matrix, columns=columns, index=columns)
    return corr_matrix

#####

def corr_cat(df,target=None,target_transform=False):
    '''
    ----------------------------------------------------------------------------------------------------------
    Función corr_cat:
    ----------------------------------------------------------------------------------------------------------
    - Funcionamiento:
        La función recibe como un dataframe, detecta las variables categóricas y calcula una 
        matriz de correlaciones mediante el uso del estadístico Cramers V. 
    - Inputs:
        -- df: DataFrame de Pandas a analizar
        -- target: String con nombre de la variable objetivo
        -- target_transform: Transforma la variable objetivo a string para el procesamiento y luego la vuelve
        a su tipo original.
    - Return:
        -- corr_cat: matriz con los Cramers V cruzados.
    '''
    df_cat_string = list(df.select_dtypes(include=['category', 'object', 'boolean']).columns.values)
    
    if target_transform:
        t_type = df[target].dtype
        df[target] = df[target].astype('string')
        df_cat_string.append(target)

    corr_cat = []
    vector = []

    for i in df_cat_string:
        vector = []
        for j in df_cat_string:
            confusion_matrix = pd.crosstab(df[i], df[j])
            vector.append(cramers_v(confusion_matrix.values))
        corr_cat.append(vector)

    corr_cat = pd.DataFrame(corr_cat, columns=df_cat_string, index=df_cat_string)
    
    if target_transform:
        df_cat_string.pop()
        df[target] = df[target].astype(t_type)

    return corr_cat



#####

def get_corr_matrix(dataset = None, metodo='pearson', size_figure=[10,8]):
    # Para obtener la correlación de Spearman, sólo cambiar el metodo por 'spearman'

    if dataset is None:
        print(u'\nHace falta pasar argumentos a la función')
        return 1
    sns.set(style="white")
    # Compute the correlation matrix
    corr = dataset.corr(method=metodo) 
    # Set self-correlation to zero to avoid distraction
    for i in range(corr.shape[0]):
        corr.iloc[i, i] = 0
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=size_figure)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, center=0,
                square=True, linewidths=.5,  cmap ='viridis' ) #cbar_kws={"shrink": .5}
    plt.show()
    
    return 0

#####
    
def double_plot(df, col_name, is_cont, target, palette=['deepskyblue','crimson'], y_scale='log'):
    """
    ----------------------------------------------------------------------------------------------------------
    Función double_plot:
    ----------------------------------------------------------------------------------------------------------
     - Funcionamiento:
        La función recibe como un dataframe y la variable a graficar. En base a si es continua o
        si es categórica, se mostrarán dos gráficos de un tipo o de otro
            - Para variables continuas se muestra un histograma y un boxplot en base al Target.
            - Para variables categóricas se muestran dos barplots, uno con la variable sola y la otra en base
            al target. Además, este segundo aplica una transformación logarítmica a la escala del eje y. Esto
            está pensado especialmente para este dataset, debido a que el desbalanceo es tan grande que casi
            no se llegan a percibir los valores 1 en la variable objetivo. Por eso para diferenciar se grafica
            de esta manera.
    - Inputs:
        -- df: DataFrame de Pandas a analizar
        -- col_name: Columna del DF a graficar
        -- is_cont: True o False. Determina si la variable a graficar es continua o no
        -- target: Variable objetivo del DF
    """
    f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,3), dpi=90)
    
    if is_cont:
        sns.histplot(df[col_name], kde=False, ax=ax1, color='limegreen')
    else:
        barplot_df = pd.DataFrame(df[col_name].value_counts()).reset_index()
        sns.barplot(barplot_df, x=col_name, y='count', palette='YlGnBu', ax=ax1)
    ax1.set_xlabel(col_name)
    ax1.set_ylabel('Count')
    ax1.set_title(col_name)
    plt.xticks(rotation = 90)

    if is_cont:
        sns.boxplot(data=df, x=col_name, y=df[target].astype('string'), palette=palette, ax=ax2)
        ax2.set_ylabel('')
        ax2.set_title(col_name + ' by '+target)
    else:
        barplot2_df = df.groupby(col_name)[target].value_counts(normalize=True).to_frame('proportion').reset_index()
        sns.barplot(data=barplot2_df, x=col_name, y='proportion', hue=barplot2_df[target].astype('string'), palette=palette, ax=ax2)
        plt.yscale(y_scale)
        ax2.set_ylabel('Proportion')       
        ax2.set_title(target)
        plt.xticks(rotation = 90)
    ax2.set_xlabel(col_name)
    
    plt.tight_layout()


######

def plot_feature(df, col_name, isContinuous, target):
    """
    Visualize a variable with and without faceting on the loan status.
    - df dataframe
    - col_name is the variable name in the dataframe
    - full_name is the full variable name
    - continuous is True if the variable is continuous, False otherwise
    """
    f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,3), dpi=90)
    
    count_null = df[col_name].isnull().sum()
    if isContinuous:
        
        sns.histplot(df.loc[df[col_name].notnull(), col_name], kde=False, ax=ax1)
    else:
        sns.countplot(df[col_name], order=sorted(df[col_name].unique()), color='#5975A4', saturation=1, ax=ax1)
    ax1.set_xlabel(col_name)
    ax1.set_ylabel('Count')
    ax1.set_title(col_name+ ' Numero de nulos: '+str(count_null))
    plt.xticks(rotation = 90)


    if isContinuous:
        sns.boxplot(x=col_name, y=target, data=df, ax=ax2)
        ax2.set_ylabel('')
        ax2.set_title(col_name + ' by '+target)
    else:
        data = df.groupby(col_name)[target].value_counts(normalize=True).to_frame('proportion').reset_index() 
        data.columns = [i, target, 'proportion'] # type: ignore
        #sns.barplot(x = col_name, y = 'proportion', hue= target, data = data, saturation=1, ax=ax2)
        sns.barplot(x = col_name, y = 'proportion', hue= target, data = data, saturation=1, ax=ax2)
        ax2.set_ylabel(target+' fraction')
        ax2.set_title(target)
        plt.xticks(rotation = 90)
    ax2.set_xlabel(col_name)
    
    plt.tight_layout()


#####

def highlight_max(s, props=''):
    """
    Función para dar formato al valor máximo de cada fila de un DataFrame.
        - s: valores a evaluar
        - props: detalle con las propiedades de estilos que se le quiere dar a la celda
    """
    return np.where(abs(s) == np.nanmax(abs(s.values)), props, '')


#####

   
def k_means_search(df, clusters_max, figsize=(6, 6)):
    """
    ----------------------------------------------------------------------------------------------------------
    Función k_means_search:
    ----------------------------------------------------------------------------------------------------------
    - Funcionamiento:
        Función que ejecuta el modelo no supervisado k-means sobre el DataFrame introducido tantas veces como
        la cantidad máxima de clusters que se quiera analizar y devuelve un gráfico que muestra la suma de
        los cuadrados de la distancia para cada cantidad de clusters.
    - Imputs:
        - df: DataFrame de Pandas sobre el que se ejecuta el K-Means
        - clusters_max: número máximo de clusters que se quiere analizar.
        - figsize: tupla con el tamaño deseado para la suma de ambos gráficos.
    """
    sse = []
    list_k = list(range(1, clusters_max+1))

    for k in list_k:
        km = KMeans(n_clusters=k)
        km.fit(df)
        sse.append(km.inertia_)

    # Plot sse against k
    plt.figure(figsize=figsize)
    plt.plot(list_k, sse, '-o')
    plt.xlabel(f'Number of clusters {k}')
    plt.ylabel('Sum of squared distance')
    plt.show()

    
#####

def feature_selection(df, add=[]):
    """
    ----------------------------------------------------------------------------------------------------------
    Función feature_selection:
    ----------------------------------------------------------------------------------------------------------
    - Funcionamiento: Recibe un DataFrame y un opcional de columnas extras a incluir. Devuelve un DataFrame
      con solo las columnas seleccionadas, además de las columnas adicionales en caso de haberse introducido.
    - Inputs:
        - df: DataFrame de Pandas del que se seleccionarán las columnas
        - add: argumento opcional en el que se pueden incluir más columnas para seleccionar
    - Return: DataFrame de pandas con las columnas seleccionadas según el feature selection aplicado.
    """
    # Columnas a seleccionar
    selected_columns = [
        'EXT_SOURCE_1', 'OCCUPATION_TYPE', 'EXT_SOURCE_3', 'EXT_SOURCE_2',
        'AMT_GOODS_PRICE', 'AMT_ANNUITY', 'DAYS_LAST_PHONE_CHANGE',
        'ORGANIZATION_TYPE', 'CODE_GENDER', 'AMT_CREDIT',
        'NAME_EDUCATION_TYPE', 'DAYS_BIRTH',
        'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH',
    ]
    
    # Si se pasan columnas adicionales, las agregamos a la lista de selección
    if add:
        selected_columns += add
    
    # Filtramos solo las columnas seleccionadas que existen en el DataFrame
    df_new = df[selected_columns]
    
    return df_new
    
##### 

def preprocessing(df, y=None, scale=True):
    """
    ----------------------------------------------------------------------------------------------------------
    Función preprocessing:
    ----------------------------------------------------------------------------------------------------------
    - Funcionamiento: Función que recibe un dataframe y realiza el preprocesamiento en base a los parámetros
      que el usuario elija. Aplica One Hot Encoding para variables con < 10 categorías, Target Encoding para
      variables con entre 10 y 20 categorías, y CatBoost Encoding para variables con > 20 categorías.
    - Inputs:
        - df: DataFrame de pandas sobre el que se generará el objeto preprocessor
        - y: la variable objetivo (opcional, pero necesaria si se usa TargetEncoding o CatBoostEncoder)
        - scale: indica si se debe aplicar StandardScaler a las variables numéricas.
    - Output: devuelve un objeto preprocessor listo para ser instanciado.
    """
    if scale:
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    else:
        numeric_transformer = Pipeline(steps=[('pass', 'passthrough')])

    # Creamos los diferentes pipelines para los encodings
    onehot_transformer = OneHotEncoder(sparse_output=False)
    mean_transformer = TargetEncoder()
    catboost_transformer = CatBoostEncoder()

    # Detectar tipos de variables
    df_bool, df_cat, df_num = tipos_vars(df, False)
    
    # Variables numéricas
    numeric_features = df.select_dtypes(include=['int64', 'float64']).drop(df_bool, axis=1).columns
    # Variables booleanas
    boolean_features = df[df_bool].columns
    
    # Variables categóricas
    categorical_features = df.select_dtypes(include=['object']).drop(df_bool, axis=1).columns
    
    # Inicializamos listas para las columnas de cada tipo de encoding
    ohe_features = []
    te_features = []
    catboost_features = []

    # Revisar las variables categóricas y asignar el tipo de encoding
    for col in categorical_features:
        n_categories = len(df[col].unique())
        if n_categories < 10:
            ohe_features.append(col)
        elif 10 <= n_categories <= 20:
            te_features.append(col)
        else:
            catboost_features.append(col)

    # Crear el preprocesador con ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('ohe', onehot_transformer, ohe_features),
            ('te', mean_transformer, te_features),
            ('catboost', catboost_transformer, catboost_features),
            ('bool', 'passthrough', boolean_features)
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    ).set_output(transform="pandas")

    return preprocessor


    
#######

def y_pred_modelo_base(y_train, X_test):
    """
    ----------------------------------------------------------------------------------------------------------
    Función y_pred_base_model_v2:
    ----------------------------------------------------------------------------------------------------------
    - Funcionamiento:
        Determina la clase mayoritaria del conjunto de entrenamiento y genera predicciones para el conjunto
        de prueba basado en probabilidades derivadas de esa clase mayoritaria.
    - Inputs:
        - y_train: Target del set de entrenamiento. DataFrame o Series.
        - X_test: DataFrame X del set de test (o validación de ser el caso).
    - Return:
        Devuelve un array de numpy con las predicciones del modelo base para los datos otorgados.
    """

    value_max = y_train.value_counts(normalize=True).idxmax()
    size = len(X_test)
    y_pred_base = np.random.choice(
        [value_max, 1 - value_max],
        size=size,
        p=[1 - value_max, value_max]
    )
    return y_pred_base


#######

def aplicar_modelos(X_train, y_train, X_test, y_test):
    modelos = {
        'Dummy': DummyClassifier(strategy='most_frequent', random_state=seed),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=seed),
        'Gradient Boosting': GradientBoostingClassifier(random_state=seed),
        'Random Forest': RandomForestClassifier(random_state=seed),
        'XGBoost': XGBClassifier(random_state=seed),
        'LightGBM': lgb.LGBMClassifier(random_state=seed),
        'AdaBoost': AdaBoostClassifier(random_state=seed),
        
    }
    
    resultados = {}
    
    for nombre, modelo in modelos.items():
        print(f"Entrenando y evaluando el modelo: {nombre}")
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        y_pred_proba = modelo.predict_proba(X_test)[:, 1] if hasattr(modelo, "predict_proba") else modelo.decision_function(X_test)
        
        # Métricas
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        cm_normalized = confusion_matrix(y_test, y_pred, normalize='true')
        
        # Curva ROC
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Curva de ganancia y Lift
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        
        resultados[nombre] = {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'classification_report': report,
            'confusion_matrix': cm,
            'confusion_matrix_normalized': cm_normalized,
            'roc_curve': (fpr, tpr, roc_auc),
            'precision_recall_curve': (precision, recall, thresholds)
        }
        
        # Mostrar métricas
        print(f"Accuracy: {accuracy}")
        print(f"ROC AUC: {roc_auc}")
        print("Classification Report:")
        print(report)
        
        # Graficar Matriz de Confusión Absoluta
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Matriz de Confusión Abs - {nombre}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Graficar Matriz de Confusión Normalizada
        plt.subplot(1, 2, 2)
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues')
        plt.title(f'Matriz de Confusión Norm - {nombre}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        plt.tight_layout()
        plt.show()
        
        # Graficar Curva ROC
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {nombre}')
        plt.legend(loc="lower right")
        plt.show()
        
        # Graficar Curva de Ganancia
        plt.figure()
        plt.plot(recall, precision, marker='.')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {nombre}')
        plt.show()
        
        # Graficar Curva Lift
        plt.figure()
        plt.plot(thresholds, precision[:-1], 'b--', label='Precision')
        plt.plot(thresholds, recall[:-1], 'g-', label='Recall')
        plt.xlabel('Threshold')
        plt.title(f'Precision-Recall vs Threshold Curve - {nombre}')
        plt.legend(loc='best')
        plt.show()
    
    return resultados

####

def roc_curve_plot_v2(y_true=None, y_pred=None, title='ROC Curve', model_name='Model', figsize=(7,5)):
    """
    ----------------------------------------------------------------------------------------------------------
    Función roc_curve_plot_v2:
    ----------------------------------------------------------------------------------------------------------
    - Imputs:
        - y_true: array/Serie con los valores reales de la variable objetivo
        - y_pred: array/Serie con las probabilidades predecidas por el modelo.
        - title: Título que se le quiera dar al gráfico.
        - model_name: Nombre del modelo implementado para mostrar en el gráfico como leyenda.
        - figsize: tupla con el tamaño deseado para el gráfico.
    """
    if ((y_true is None) or (y_pred is None)):
        print(u'\nFaltan parámetros por pasar a la función')
        return 1

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    gmeans = np.sqrt(tpr * (1-fpr))
    ix = np.argmax(gmeans)
    print('Best Threshold = %f, G-Mean = %.3f' % (thresholds[ix], gmeans[ix]))
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, marker='o', color='orange', lw=2, label=f'{model_name} (area = %0.3f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='blue', lw=3, linestyle='--', label='No Skill')
    plt.scatter(fpr[ix], tpr[ix], s=100, marker='o', color='black', label='Best', zorder=2)
    ax.set_xlim([-0.025, 1.025])
    ax.set_ylim([-0.025, 1.025])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title, fontdict={'fontsize':18})
    ax.legend(loc="lower right")
    ax.grid(alpha=0.5)

    gini = (2.0 * roc_auc) - 1.0

    print('\n*************************************************************')
    print(u'\nEl coeficiente de GINI es: %0.2f' % gini)
    print(u'\nEl área por debajo de la curva ROC es: %0.4f' %roc_auc)
    print('\n*************************************************************')
    
####

def pr_curve_plot(y_true, y_pred_proba, title='Precision-Recall Curve', f_score_beta=1, model_name='Model', figsize=(7,5)):
    """
    ----------------------------------------------------------------------------------------------------------
    Función pr_curve_plot:
    ----------------------------------------------------------------------------------------------------------
    - Imputs:
        - y_true: array/Serie con los valores reales de la variable objetivo (y_test o y_val)
        - y_pred_proba: array/Serie con las probabilidades predecidas por el modelo.
        - title: Título que se le quiera dar al gráfico.
        - f_score_beta: Beta para el F score. Normalmente 0.5, 1 o 2.
        - model_name: Nombre del modelo implementado para mostrar en el gráfico como leyenda.
        - figsize: tupla con el tamaño deseado para el gráfico.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)

    f_score = ((1+(f_score_beta**2)) * precision * recall) / ((f_score_beta**2) * precision + recall)
    ix = np.argmax(f_score)
    auc_rp = auc(recall, precision)
    print(f'Best Threshold = {thresholds[ix]:.5f}, F{f_score_beta} Score = {f_score[ix]:.3f}, AUC = {auc_rp:.4f}')
    
    #plt.ylim([0,1])
    fig, ax = plt.subplots(figsize=figsize)
    no_skill= len(y_true[y_true==1])/len(y_true)
    ax.plot([0,1],[no_skill, no_skill], linestyle='--', label='No Skill', color='dodgerblue', lw=3)
    ax.plot(recall, precision, marker='.', label=model_name, color='orange')
    ax.scatter(recall[ix], precision[ix], s=100, marker='o', color='black', label=f'Best', zorder=2)
    ax.set_title(str(title), fontdict={'fontsize':18})
    ax.set_ylim([0,1.01])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend()
    ax.grid(alpha=0.5)
    
########

def summarize_metrics(y_true, y_pred):
    """
    ----------------------------------------------------------------------------------------------------------
    Función summarize_metrics:
    ----------------------------------------------------------------------------------------------------------
    - Funcionamiento:
        Calcula y muestra varias métricas de evaluación para un modelo de clasificación. Las métricas calculadas son:
            - Accuracy
            - Balanced Accuracy
            - F2 Score
            - F1 Score
            - Precision
            - Recall
            - Confusion Matrix
    - Inputs:
        - y_true: array/Serie con los valores reales de la variable target (e.g., y_test o y_val).
        - y_pred: array/Serie con los valores predichos por el modelo.
    """
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, fbeta_score, f1_score,
        precision_score, recall_score, confusion_matrix
    )

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "F2 Score": fbeta_score(y_true, y_pred, beta=2),
        "F1 Score": f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred)
    }
    
    # Mostrar métricas
    for metric, value in metrics.items():
        print(f"{metric}: {value:.5f}")
    
    # Mostrar matriz de confusión
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


########

def plot_recall_precision(recall_precision, y_true, y_pred_proba):
    """
    ----------------------------------------------------------------------------------------------------------
    Función plot_recall_precision:
    ----------------------------------------------------------------------------------------------------------
    - Funcionamiento:
        Función para graficar las métricas del modelo (Precision, Recall, F2 Score, F1 Score) en función
        del threshold, y mostrar el threshold óptimo que maximiza el F2 Score.
    - Inputs:
        - recall_precision: Lista de listas en las que cada elemento contiene un threshold y las 4 métricas:
                             threshold, Recall, Precision, F2 Score, F1 Score.
        - y_true: Valores reales de la variable objetivo.
        - y_pred_proba: Probabilidades predichas por el modelo.
    """
    # Extraer las métricas de recall_precision
    thresholds = [round(item[0], 2) for item in recall_precision]
    recall = [item[1] for item in recall_precision]
    precision = [item[2] for item in recall_precision]
    f2_score = [item[3] for item in recall_precision]
    f1_score = [item[4] for item in recall_precision]

    # Crear la figura
    plt.figure(figsize=(15, 5))

    # Graficar las métricas
    sns.lineplot(x=thresholds, y=recall, color="red", label='Recall', marker='o')
    sns.lineplot(x=thresholds, y=precision, color="blue", label='Precision', marker='o')
    sns.lineplot(x=thresholds, y=f2_score, color="gold", label='F2 Score', marker='o')
    sns.lineplot(x=thresholds, y=f1_score, color="limegreen", label='F1 Score', marker='o')

    # Calcular y graficar el F2 Score
    precision_vals, recall_vals, thresholds_vals = precision_recall_curve(y_true, y_pred_proba)
    f2_vals = (5 * precision_vals * recall_vals) / (4 * precision_vals + recall_vals)
    best_f2_idx = np.argmax(f2_vals)

    # Añadir el punto óptimo del F2 Score en el gráfico
    plt.scatter(thresholds_vals[best_f2_idx], f2_vals[best_f2_idx], s=100, marker='o', color='black',
                label=f'Best F2 (th={thresholds_vals[best_f2_idx]:.3f}, f2={f2_vals[best_f2_idx]:.3f})', zorder=5)

    # Ajustes y etiquetas
    plt.title('Recall & Precision VS Threshold', fontsize=20)
    plt.xlabel('Threshold', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.legend()

    # Ajustar etiquetas del eje x
    plt.xticks(rotation=45, fontsize=10)

    # Mostrar el gráfico
    plt.tight_layout()
    plt.show()
#####


def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix', figsize=(20, 6)):
    """
    ----------------------------------------------------------------------------------------------------------
    Función plot_confusion_matrix:
    ----------------------------------------------------------------------------------------------------------
    - Funcionamiento:
        Grafica la matriz de confusión en dos formatos: valores absolutos y normalizados.
    - Inputs:
        - y_true: array/Serie con los valores reales de la variable objetivo (y_test o y_val).
        - y_pred: array/Serie con las predicciones realizadas por el modelo.
        - title: Título del gráfico.
        - figsize: Tupla que define el tamaño del gráfico.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    from matplotlib import rc

    # Crear la figura con dos subgráficos
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Matriz de confusión absoluta
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, cmap='Blues', values_format=',.0f', ax=axes[0]
    )
    axes[0].set_title(f'{title}', fontdict={'fontsize': 18})
    axes[0].set_xlabel('Predicted Label', fontdict={'fontsize': 15})
    axes[0].set_ylabel('True Label', fontdict={'fontsize': 15})

    # Matriz de confusión normalizada
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, cmap='Blues', normalize='true', values_format='.2%', ax=axes[1]
    )
    axes[1].set_title(f'{title} - Normalized', fontdict={'fontsize': 18})
    axes[1].set_xlabel('Predicted Label', fontdict={'fontsize': 15})
    axes[1].set_ylabel('True Label', fontdict={'fontsize': 15})

    # Configuración general de la fuente
    rc('font', size=14)
    rc('xtick', labelsize=12)
    rc('ytick', labelsize=12)

    # Mostrar los gráficos
    plt.tight_layout()
    plt.show()

####

def plot_cumulative_gain(y_true, y_pred_proba, figsize=(7, 5)):
    """
    Función para graficar el Cumulative Gain Chart sin usar scikit-plot.
    """
    # Ordenar los datos por probabilidad predicha, de mayor a menor
    sorted_indices = np.argsort(y_pred_proba)[::-1]
    y_true_sorted = np.array(y_true)[sorted_indices]
    
    # Calcular las tasas acumulativas
    total_positive = np.sum(y_true_sorted)
    total_negative = len(y_true_sorted) - total_positive
    
    cumulative_positive = np.cumsum(y_true_sorted)
    cumulative_gain = cumulative_positive / total_positive
    
    cumulative_percentage = np.arange(1, len(y_true_sorted) + 1) / len(y_true_sorted)
    
    # Crear la gráfica
    plt.figure(figsize=figsize)
    plt.plot(cumulative_percentage, cumulative_gain, label='Cumulative Gain', color='orange', lw=2)
    plt.plot(cumulative_percentage, cumulative_percentage, label='Baseline (Random)', color='blue', linestyle='--', lw=2)
    plt.title('Cumulative Gain Chart', fontsize=14)
    plt.xlabel('Percentage of Samples', fontsize=12)
    plt.ylabel('Gain (Percentage of Positives)', fontsize=12)
    plt.xlim([-0.025, 1.025])
    plt.ylim([-0.025, 1.025])
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.show()


####


def plot_lift_curve(y_true, y_pred_proba, ax=None, figsize=(10, 6)):
    """
    Plot the Lift Curve.

    Parameters:
    y_true (array-like): True binary labels.
    y_pred_proba (array-like): Target scores, can either be probability estimates of the positive class.
    ax (matplotlib.axes.Axes, optional): Axes object to draw the plot onto, otherwise uses the current Axes.
    figsize (tuple, optional): Figure size.

    Returns:
    matplotlib.axes.Axes: Axes object with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Sort by predicted probabilities
    sorted_indices = np.argsort(y_pred_proba)[::-1]
    y_true_sorted = y_true[sorted_indices]

    # Calculate cumulative gains
    cumulative_gains = np.cumsum(y_true_sorted) / np.sum(y_true_sorted)

    # Calculate lift
    lift = cumulative_gains / (np.arange(1, len(y_true_sorted) + 1) / len(y_true_sorted))

    # Plot lift curve
    ax.plot(np.arange(1, len(y_true_sorted) + 1) / len(y_true_sorted), lift, label='Lift Curve', color='orange')
    ax.plot([0, 1], [1, 1], 'k--', label='Baseline', color='blue')

    # Set plot labels and title
    ax.set_xlabel('Percentage of Sample')
    ax.set_ylabel('Lift')
    ax.set_title('Lift Curve')
    ax.legend()

    return ax


####

# Destacar que muchas de las funciones programadas aquí fueron inspiradas en contenidos vistos en el máster y en internet, sobre todo libros y diversos proyectos de machine learning
# En ningún momento se realizó plagio, las funciones que adquirí de diversas fuentes fueron reformuladas y adaptadas a mi caso de estudio, además citaré la bibliografía consultada a continuación

# BIBLIOGRAFÍA #

# html y plantillas proporcionados por los profesores
# https://github.com/data-flair/machine-learning-projects?tab=readme-ov-file
# https://github.com/ashishpatel26/500-AI-Machine-learning-Deep-learning-Computer-vision-NLP-Projects-with-code?tab=readme-ov-file
# https://fedefliguer.github.io/AAI/
# https://github.com/rodrifer10
# https://cienciadedatos.net/documentos/py06_machine_learning_python_scikitlearn
