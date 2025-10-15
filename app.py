import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, OrdinalEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    LinearRegression, LogisticRegression,
    Ridge, Lasso
)
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    mean_squared_error, r2_score, 
    accuracy_score, classification_report, confusion_matrix
)
import pickle
from PIL import Image

# Configuración de la página
st.set_page_config(
    page_title="Aprendizaje Supervisado",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Directorios de la aplicación
UPLOAD_DIR = "uploads"
MODELS_DIR = "models"

# Crear directorios si no existen
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Inicializar variables de sesión
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'preprocessed_data' not in st.session_state:
    st.session_state.preprocessed_data = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_type' not in st.session_state:
    st.session_state.model_type = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'problem_type' not in st.session_state:
    st.session_state.problem_type = None

# Función para cargar datos
def load_data(file):
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
        return None

# Función para guardar datos
def save_data(df, filename):
    try:
        filepath = os.path.join(UPLOAD_DIR, filename)
        df.to_csv(filepath, index=False)
        return filepath
    except Exception as e:
        st.error(f"Error al guardar el archivo: {e}")
        return None

# Función para guardar modelo y preprocesador
def save_model(model, preprocessor, model_info, filename):
    try:
        model_path = os.path.join(MODELS_DIR, f"{filename}.pkl")
        preproc_path = os.path.join(MODELS_DIR, f"{filename}_preproc.pkl")
        info_path = os.path.join(MODELS_DIR, f"{filename}.json")
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        with open(preproc_path, 'wb') as f:
            pickle.dump(preprocessor, f)
        with open(info_path, 'w') as f:
            json.dump(model_info, f)
            
        return model_path
    except Exception as e:
        st.error(f"Error al guardar el modelo: {e}")
        return None

# Función para cargar modelo y preprocesador
def load_model(filename):
    try:
        model_path = os.path.join(MODELS_DIR, f"{filename}.pkl")
        preproc_path = os.path.join(MODELS_DIR, f"{filename}_preproc.pkl")
        info_path = os.path.join(MODELS_DIR, f"{filename}.json")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(preproc_path, 'rb') as f:
            preprocessor = pickle.load(f)
        with open(info_path, 'r') as f:
            model_info = json.load(f)
            
        return model, preprocessor, model_info
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None, None, None

# Header personalizado con imágenes y texto de la universidad
def custom_header():
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if os.path.exists("una puno.jpeg"):
            st.image("una puno.jpeg", width=150)
    
    with col2:
        st.markdown("<h2 style='text-align: center;'>UNIVERSIDAD NACIONAL DEL ALTIPLANO-UNA PUNO</h2>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center;'>ESCUELA PROFESIONAL DE INGENIERIA ESTADISTICA E INFORMATICA</h3>", unsafe_allow_html=True)
    
    with col3:
        if os.path.exists("estadistica.jpeg"):
            st.image("estadistica.jpeg", width=150)
    
    st.markdown("---")

# Mostrar el header personalizado en todas las páginas
custom_header()

# Sidebar para navegación
st.sidebar.title("Navegación")
page = st.sidebar.radio(
    "Selecciona una página",
    ["Inicio", "Cargar Datos", "Preprocesar Datos", "Entrenar Modelo", "Evaluar Modelo", "Predicción"]
)

# Página de inicio
if page == "Inicio":
    st.title("Aprendizaje Supervisado")
    st.write("Bienvenido a la aplicación de Aprendizaje Supervisado. Esta herramienta te permite cargar datos, preprocesarlos, entrenar modelos de machine learning, evaluarlos y hacer predicciones.")
    
    st.header("Características principales:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📊 Datos")
        st.write("- Carga y gestión de datos")
        st.write("- Eliminar o imputar valores faltantes")
        st.write("- Selección de columnas relevantes")
    
    with col2:
        st.subheader("🧠 Modelos")
        st.write("- Regresión Lineal / Logística")
        st.write("- Ridge, Lasso, Árboles, KNN")
        st.write("- Codificación categórica (0/1 o One-Hot)")
    
    with col3:
        st.subheader("📥 Descargas")
        st.write("- Datos preprocesados (CSV)")
        st.write("- Métricas del modelo (CSV)")
        st.write("- Predicciones en nuevos datos (CSV)")
    
    st.info("👈 Utiliza el menú de navegación para comenzar tu proyecto de aprendizaje automático.")

# Página de carga de datos
elif page == "Cargar Datos":
    st.title("Carga de Datos")
    st.write("Sube un archivo CSV con tus datos para comenzar el análisis.")
    
    uploaded_file = st.file_uploader("Selecciona un archivo CSV", type=["csv"])
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            dataset_name = st.text_input("Nombre del conjunto de datos")
            dataset_description = st.text_area("Descripción (opcional)")
        
        with col2:
            df = load_data(uploaded_file)
            if df is not None:
                st.write(f"Columnas disponibles: {', '.join(df.columns)}")
                target_column = st.selectbox("Selecciona la variable objetivo (target)", df.columns)
        
        if st.button("Cargar Datos") and dataset_name and target_column:
            if df is not None:
                filepath = save_data(df, f"{dataset_name}.csv")
                if filepath:
                    st.session_state.dataset = df
                    st.session_state.target_column = target_column
                    st.success(f"Datos cargados correctamente: {len(df)} filas y {len(df.columns)} columnas")
                    st.subheader("Vista previa de los datos")
                    st.dataframe(df.head())
                    st.subheader("Información básica")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Número de filas:** {df.shape[0]}")
                        st.write(f"**Número de columnas:** {df.shape[1]}")
                        st.write(f"**Variable objetivo:** {target_column}")
                    with col2:
                        missing_values = df.isnull().sum().sum()
                        st.write(f"**Valores faltantes:** {missing_values}")
                        st.write(f"**Tipos de datos:**")
                        st.write(df.dtypes)
    
    # Mostrar datasets guardados
    st.subheader("Conjuntos de datos guardados")
    saved_datasets = [f for f in os.listdir(UPLOAD_DIR) if f.endswith('.csv')]
    if saved_datasets:
        selected_dataset = st.selectbox("Selecciona un conjunto de datos guardado", saved_datasets)
        if st.button("Cargar dataset guardado"):
            df = pd.read_csv(os.path.join(UPLOAD_DIR, selected_dataset))
            st.session_state.dataset = df
            st.success(f"Dataset {selected_dataset} cargado correctamente")
            st.dataframe(df.head())
    else:
        st.info("No hay conjuntos de datos guardados")

# Página de preprocesamiento
elif page == "Preprocesar Datos":
    st.title("Preprocesamiento de Datos")
    
    if st.session_state.dataset is None:
        st.warning("No hay datos cargados. Por favor, carga un conjunto de datos primero.")
        st.info("Ve al menú lateral y selecciona 'Cargar Datos'.")
    else:
        df = st.session_state.dataset
        target = st.session_state.target_column
        
        st.subheader("Resumen del conjunto de datos")
        st.write(f"**Dimensiones:** {df.shape[0]} filas x {df.shape[1]} columnas")
        st.write(f"**Variable objetivo:** {target}")
        
        col_info = pd.DataFrame({
            'Tipo': df.dtypes,
            'No Nulos': df.count(),
            'Nulos': df.isnull().sum(),
            '% Nulos': (df.isnull().sum() / len(df) * 100).round(2),
            'Únicos': df.nunique()
        })
        st.dataframe(col_info)
        
        features_to_keep = st.multiselect(
            "Selecciona las columnas a mantener (elimina las no seleccionadas)",
            [col for col in df.columns if col != target],
            default=[col for col in df.columns if col != target]
        )
        st.info(f"Se mantendrán {len(features_to_keep)} columnas. Las demás serán eliminadas.")
        
        # Manejo de valores faltantes
        st.write("**2. Manejo de valores faltantes**")
        handle_missing = st.radio(
            "¿Cómo manejar los valores faltantes?",
            ["Eliminar filas con valores faltantes", "Imputar valores faltantes"],
            index=1
        )

        numeric_cols = df[features_to_keep].select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df[features_to_keep].select_dtypes(include=['object', 'category']).columns.tolist()

        if handle_missing == "Imputar valores faltantes":
            numeric_imputer = st.selectbox(
                "Estrategia para valores faltantes numéricos",
                ["media", "mediana", "más frecuente", "constante"],
                index=0
            )
            
            categorical_imputer = st.selectbox(
                "Estrategia para valores faltantes categóricos",
                ["más frecuente", "constante"],
                index=0
            )
        else:
            numeric_imputer = None
            categorical_imputer = None
        
        # Codificación de variables categóricas
        st.write("**3. Codificación de variables categóricas**")
        if categorical_cols:
            # Verificar si alguna columna categórica es binaria
            binary_cats = [col for col in categorical_cols if df[col].nunique() == 2]
            if binary_cats:
                st.write(f"Variables binarias detectadas: {', '.join(binary_cats)}")
            encoding_method = st.selectbox(
                "Método de codificación",
                ["One-Hot Encoding (todas las categorías)", "Codificación binaria (0/1) para 2 categorías"],
                index=0
            )
        else:
            encoding_method = "Ninguna"
            st.info("No hay variables categóricas.")
        
        # Escalado de características
        st.write("**4. Escalado de características**")
        scaling_method = st.selectbox(
            "Método de escalado para variables numéricas",
            ["Ninguno", "Estandarización", "Normalización Min-Max"],
            index=0
        )
        
        # División de datos
        st.write("**5. División de datos**")
        test_size = st.slider("Porcentaje de datos para prueba", 10, 40, 20) / 100
        random_state = st.number_input("Semilla aleatoria", value=42)
        
        if st.button("Aplicar Preprocesamiento"):
            try:
                X = df[features_to_keep]
                y = df[target]

                # Manejo de valores faltantes
                if handle_missing == "Eliminar filas con valores faltantes":
                    df_combined = pd.concat([X, y], axis=1).dropna()
                    X = df_combined[features_to_keep]
                    y = df_combined[target]
                    st.info(f"✅ Se eliminaron {len(df) - len(df_combined)} filas con valores faltantes.")
                # Si se imputa, se hará en el pipeline

                # Dividir los datos
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
                
                # Crear transformadores
                numeric_transformer = Pipeline(steps=[])
                categorical_transformer = Pipeline(steps=[])
                
                # Imputación (solo si se eligió imputar)
                if handle_missing == "Imputar valores faltantes":
                    if numeric_imputer == "media":
                        numeric_transformer.steps.append(('imputer', SimpleImputer(strategy='mean')))
                    elif numeric_imputer == "mediana":
                        numeric_transformer.steps.append(('imputer', SimpleImputer(strategy='median')))
                    elif numeric_imputer == "más frecuente":
                        numeric_transformer.steps.append(('imputer', SimpleImputer(strategy='most_frequent')))
                    elif numeric_imputer == "constante":
                        numeric_transformer.steps.append(('imputer', SimpleImputer(strategy='constant', fill_value=0)))
                    
                    if categorical_imputer == "más frecuente":
                        categorical_transformer.steps.append(('imputer', SimpleImputer(strategy='most_frequent')))
                    elif categorical_imputer == "constante":
                        categorical_transformer.steps.append(('imputer', SimpleImputer(strategy='constant', fill_value='missing')))
                
                # Escalado
                if scaling_method == "Estandarización":
                    numeric_transformer.steps.append(('scaler', StandardScaler()))
                elif scaling_method == "Normalización Min-Max":
                    numeric_transformer.steps.append(('scaler', MinMaxScaler()))
                
                # Codificación
                if encoding_method == "One-Hot Encoding (todas las categorías)":
                    categorical_transformer.steps.append(('encoder', OneHotEncoder(handle_unknown='ignore')))
                elif encoding_method == "Codificación binaria (0/1) para 2 categorías":
                    categorical_transformer.steps.append(('encoder', OrdinalEncoder()))
                
                # Preprocesador
                transformers = []
                if numeric_cols:
                    transformers.append(('num', numeric_transformer, numeric_cols))
                if categorical_cols:
                    transformers.append(('cat', categorical_transformer, categorical_cols))
                
                if not transformers:
                    # Si no hay columnas numéricas ni categóricas, usar passthrough
                    preprocessor = FunctionTransformer(lambda x: x, validate=False)
                else:
                    preprocessor = ColumnTransformer(transformers, remainder='passthrough')
                
                # Aplicar preprocesamiento
                if not transformers:
                    X_train_processed = X_train.values
                    X_test_processed = X_test.values
                else:
                    X_train_processed = preprocessor.fit_transform(X_train)
                    X_test_processed = preprocessor.transform(X_test)
                
                st.session_state.preprocessed_data = {
                    'X_train': X_train_processed,
                    'X_test': X_test_processed,
                    'y_train': y_train,
                    'y_test': y_test,
                    'preprocessor': preprocessor,
                    'feature_names': features_to_keep,
                    'target': target,
                    'preprocessing_info': {
                        'numeric_cols': numeric_cols,
                        'categorical_cols': categorical_cols,
                        'handle_missing': handle_missing,
                        'numeric_imputer': numeric_imputer,
                        'categorical_imputer': categorical_imputer,
                        'encoding_method': encoding_method,
                        'scaling_method': scaling_method,
                        'test_size': test_size,
                        'random_state': random_state
                    }
                }
                
                st.success("✅ Preprocesamiento aplicado correctamente")
                st.write(f"**Conjunto de entrenamiento:** {X_train_processed.shape[0]} muestras")
                st.write(f"**Conjunto de prueba:** {X_test_processed.shape[0]} muestras")
                
                # Determinar tipo de problema
                if y.dtype == 'object' or y.nunique() < 10:
                    problem_type = "Clasificación"
                    st.write(f"**Tipo de problema:** {problem_type}")
                    st.write(f"**Clases:** {y.nunique()}")
                    st.write(y.value_counts())
                else:
                    problem_type = "Regresión"
                    st.write(f"**Tipo de problema:** {problem_type}")
                    st.write(y.describe())
                
                st.session_state.problem_type = problem_type
                
            except Exception as e:
                st.error(f"Error durante el preprocesamiento: {e}")

# Página de entrenamiento del modelo
elif page == "Entrenar Modelo":
    st.title("Entrenamiento del Modelo")
    
    if st.session_state.preprocessed_data is None:
        st.warning("No hay datos preprocesados. Por favor, preprocesa los datos primero.")
        st.info("Ve al menú lateral y selecciona 'Preprocesar Datos'.")
    else:
        preprocessed_data = st.session_state.preprocessed_data
        problem_type = st.session_state.problem_type
        
        st.subheader("Información de los datos preprocesados")
        st.write(f"**Conjunto de entrenamiento:** {preprocessed_data['X_train'].shape[0]} muestras")
        st.write(f"**Variable objetivo:** {preprocessed_data['target']}")
        st.write(f"**Tipo de problema:** {problem_type}")
        
        # Selección del modelo
        if problem_type == "Clasificación":
            model_type = st.selectbox(
                "Selecciona un modelo de clasificación",
                [
                    "Regresión Logística",
                    "Árbol de Decisión (CART)",
                    "K-Nearest Neighbors (KNN)",
                    "Random Forest",
                    "Red Neuronal (MLP)",
                    "SVM",
                    "Gradient Boosting"
                ]
            )
        else:
            model_type = st.selectbox(
                "Selecciona un modelo de regresión",
                [
                    "Regresión Lineal",
                    "Ridge",
                    "Lasso",
                    "Árbol de Decisión (CART)",
                    "K-Nearest Neighbors (KNN)",
                    "Random Forest",
                    "Red Neuronal (MLP)",
                    "SVR",
                    "Gradient Boosting"
                ]
            )
        
        # Hiperparámetros
        hyperparams = {}
        
        if model_type in ["Random Forest", "Gradient Boosting"]:
            hyperparams['n_estimators'] = st.slider("Número de estimadores", 10, 500, 100)
            hyperparams['max_depth'] = st.slider("Profundidad máxima", 1, 30, 10)
        
        elif model_type in ["Árbol de Decisión (CART)"]:
            hyperparams['max_depth'] = st.slider("Profundidad máxima", 1, 30, 5)
            hyperparams['random_state'] = 42
        
        elif model_type in ["K-Nearest Neighbors (KNN)"]:
            hyperparams['n_neighbors'] = st.slider("Número de vecinos (k)", 1, 20, 5)
        
        elif model_type in ["Red Neuronal (MLP)"]:
            hyperparams['hidden_layer_sizes'] = (100,)
            hyperparams['max_iter'] = 1000
            hyperparams['random_state'] = 42
        
        elif model_type in ["Regresión Logística", "Regresión Lineal", "Ridge", "Lasso"]:
            hyperparams['fit_intercept'] = st.checkbox("Ajustar intercepto", True)
            if model_type in ["Ridge", "Lasso"]:
                hyperparams['alpha'] = st.slider("Regularización (alpha)", 0.01, 10.0, 1.0)
        
        elif model_type in ["SVM", "SVR"]:
            hyperparams['C'] = st.slider("Parámetro de regularización (C)", 0.1, 10.0, 1.0)
            hyperparams['kernel'] = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"], index=2)
        
        # Opción para guardar
        save_to_disk = st.checkbox("Guardar modelo en disco", value=True)
        model_name = st.text_input(
            "Nombre para guardar el modelo",
            value=f"{model_type.lower().replace(' ', '_').replace('(', '').replace(')', '')}_{preprocessed_data['target']}"
        )
        
        if st.button("Entrenar Modelo"):
            try:
                X_train = preprocessed_data['X_train']
                y_train = preprocessed_data['y_train']
                
                # Crear modelo según selección
                if model_type == "Regresión Lineal":
                    model = LinearRegression(fit_intercept=hyperparams.get('fit_intercept', True))
                
                elif model_type == "Regresión Logística":
                    model = LogisticRegression(
                        fit_intercept=hyperparams.get('fit_intercept', True),
                        max_iter=1000,
                        random_state=42
                    )
                
                elif model_type == "Ridge":
                    model = Ridge(
                        alpha=hyperparams.get('alpha', 1.0),
                        fit_intercept=hyperparams.get('fit_intercept', True),
                        random_state=42
                    )
                
                elif model_type == "Lasso":
                    model = Lasso(
                        alpha=hyperparams.get('alpha', 1.0),
                        fit_intercept=hyperparams.get('fit_intercept', True),
                        random_state=42,
                        max_iter=10000
                    )
                
                elif model_type == "Árbol de Decisión (CART)" and problem_type == "Clasificación":
                    model = DecisionTreeClassifier(
                        max_depth=hyperparams.get('max_depth', 5),
                        random_state=42
                    )
                elif model_type == "Árbol de Decisión (CART)" and problem_type == "Regresión":
                    model = DecisionTreeRegressor(
                        max_depth=hyperparams.get('max_depth', 5),
                        random_state=42
                    )
                
                elif model_type == "K-Nearest Neighbors (KNN)" and problem_type == "Clasificación":
                    model = KNeighborsClassifier(n_neighbors=hyperparams.get('n_neighbors', 5))
                elif model_type == "K-Nearest Neighbors (KNN)" and problem_type == "Regresión":
                    model = KNeighborsRegressor(n_neighbors=hyperparams.get('n_neighbors', 5))
                
                elif model_type == "Red Neuronal (MLP)" and problem_type == "Clasificación":
                    model = MLPClassifier(
                        hidden_layer_sizes=hyperparams.get('hidden_layer_sizes', (100,)),
                        max_iter=hyperparams.get('max_iter', 1000),
                        random_state=42
                    )
                elif model_type == "Red Neuronal (MLP)" and problem_type == "Regresión":
                    model = MLPRegressor(
                        hidden_layer_sizes=hyperparams.get('hidden_layer_sizes', (100,)),
                        max_iter=hyperparams.get('max_iter', 1000),
                        random_state=42
                    )
                
                elif model_type == "SVM":
                    model = SVC(
                        C=hyperparams.get('C', 1.0),
                        kernel=hyperparams.get('kernel', 'rbf'),
                        probability=True,
                        random_state=42
                    )
                elif model_type == "SVR":
                    model = SVR(
                        C=hyperparams.get('C', 1.0),
                        kernel=hyperparams.get('kernel', 'rbf')
                    )
                
                elif model_type == "Gradient Boosting" and problem_type == "Clasificación":
                    model = GradientBoostingClassifier(
                        n_estimators=hyperparams.get('n_estimators', 100),
                        max_depth=hyperparams.get('max_depth', 3),
                        random_state=42
                    )
                elif model_type == "Gradient Boosting" and problem_type == "Regresión":
                    model = GradientBoostingRegressor(
                        n_estimators=hyperparams.get('n_estimators', 100),
                        max_depth=hyperparams.get('max_depth', 3),
                        random_state=42
                    )
                
                else:
                    st.error("Modelo no soportado.")
                    st.stop()

                # Entrenar el modelo
                with st.spinner(f"Entrenando {model_type}..."):
                    model.fit(X_train, y_train)
                
                st.session_state.model = model
                st.session_state.model_type = model_type
                
                model_info = {
                    'model_type': model_type,
                    'hyperparams': hyperparams,
                    'problem_type': problem_type,
                    'target': preprocessed_data['target'],
                    'feature_names': preprocessed_data['feature_names'],
                    'preprocessing_info': preprocessed_data['preprocessing_info']
                }
                
                if save_to_disk and model_name:
                    save_model(model, preprocessed_data['preprocessor'], model_info, model_name)
                    st.success(f"Modelo guardado como '{model_name}'")
                
                st.success("✅ Modelo entrenado correctamente. Ahora puedes evaluarlo.")
                
                # Si es un árbol, mostrar el gráfico
                if "Árbol de Decisión" in model_type:
                    st.subheader("Visualización del Árbol de Decisión")
                    
                    # Obtener nombres reales tras preprocesamiento
                    try:
                        feature_names = preprocessed_data['preprocessor'].get_feature_names_out()
                    except Exception as e:
                        n_features = X_train.shape[1]
                        feature_names = [f"feature_{i}" for i in range(n_features)]
                    
                    fig, ax = plt.subplots(figsize=(15, 8))
                    plot_tree(
                        model,
                        feature_names=feature_names,
                        filled=True,
                        rounded=True,
                        fontsize=8,
                        ax=ax
                    )
                    st.pyplot(fig)
                    
                    if problem_type == "Regresión":
                        st.write("📌 El gráfico muestra:")
                        st.write("- `squared_error`: error cuadrático dentro del nodo")
                        st.write("- `samples`: número de muestras en el nodo")
                        st.write("- `value`: valor promedio de la variable objetivo en el nodo")
                
            except Exception as e:
                st.error(f"Error durante el entrenamiento: {e}")

# Página de evaluación
elif page == "Evaluar Modelo":
    st.title("Evaluación del Modelo")
    
    if st.session_state.model is None:
        st.warning("No hay modelo entrenado. Por favor, entrena un modelo primero.")
        st.info("Ve al menú lateral y selecciona 'Entrenar Modelo'.")
    else:
        model = st.session_state.model
        model_type = st.session_state.model_type
        preprocessed_data = st.session_state.preprocessed_data
        problem_type = st.session_state.problem_type
        
        X_test = preprocessed_data['X_test']
        y_test = preprocessed_data['y_test']
        y_pred = model.predict(X_test)
        
        st.subheader("Métricas de rendimiento")
        
        if problem_type == "Clasificación":
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"**Exactitud (Accuracy):** {accuracy:.4f}")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())
            
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title('Matriz de Confusión')
            st.pyplot(fig)
            
            st.session_state.metrics = {'accuracy': accuracy, 'classification_report': report}
            
        else:
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            st.write(f"**MSE:** {mse:.4f} | **RMSE:** {rmse:.4f} | **R²:** {r2:.4f}")
            
            fig, ax = plt.subplots(1, 2, figsize=(14, 5))
            ax[0].scatter(y_test, y_pred, alpha=0.6)
            ax[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            ax[0].set_xlabel('Valores reales')
            ax[0].set_ylabel('Predicciones')
            ax[0].set_title('Valores reales vs Predicciones')
            
            residuals = y_test - y_pred
            ax[1].scatter(y_pred, residuals, alpha=0.6)
            ax[1].axhline(0, color='r', linestyle='--')
            ax[1].set_xlabel('Predicciones')
            ax[1].set_ylabel('Residuos')
            ax[1].set_title('Residuos')
            st.pyplot(fig)
            
            st.session_state.metrics = {'mse': mse, 'rmse': rmse, 'r2': r2}
        
        # Importancia de características
        if hasattr(model, 'feature_importances_'):
            st.subheader("Importancia de características")
            try:
                feature_names = preprocessed_data['preprocessor'].get_feature_names_out()
            except:
                feature_names = [f"feature_{i}" for i in range(len(model.feature_importances_))]
            
            importances = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(20)
            
            st.dataframe(importances)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='importance', y='feature', data=importances, ax=ax)
            st.pyplot(fig)
        
        # 📥 Botones de descarga
        st.subheader("📥 Descargar Resultados")
        
        # Descargar datos de entrenamiento preprocesados
        try:
            feature_names_out = preprocessed_data['preprocessor'].get_feature_names_out()
        except:
            feature_names_out = [f"feature_{i}" for i in range(preprocessed_data['X_train'].shape[1])]
        
        X_train_df = pd.DataFrame(preprocessed_data['X_train'], columns=feature_names_out)
        y_train_df = pd.Series(preprocessed_data['y_train'], name=preprocessed_data['target'])
        train_full = pd.concat([X_train_df, y_train_df], axis=1)
        
        st.download_button(
            label="⬇️ Descargar datos de entrenamiento (CSV)",
            data=train_full.to_csv(index=False),
            file_name="datos_entrenamiento_preprocesados.csv",
            mime="text/csv"
        )
        
        # Descargar métricas
        if st.session_state.metrics:
            metrics_df = pd.DataFrame([st.session_state.metrics])
            st.download_button(
                label="⬇️ Descargar métricas del modelo (CSV)",
                data=metrics_df.to_csv(index=False),
                file_name="metricas_modelo.csv",
                mime="text/csv"
            )

# Página de predicción
elif page == "Predicción":
    st.title("Predicción en Nuevos Datos")
    
    model_option = st.radio("¿Qué modelo deseas utilizar?", ["Modelo actual", "Cargar modelo guardado"])
    
    model = preprocessor = model_info = None
    
    if model_option == "Modelo actual":
        if st.session_state.model is None:
            st.warning("No hay modelo actual. Entrena uno o carga uno guardado.")
        else:
            model = st.session_state.model
            preprocessor = st.session_state.preprocessed_data['preprocessor']
            model_info = {
                'model_type': st.session_state.model_type,
                'problem_type': st.session_state.problem_type,
                'target': st.session_state.preprocessed_data['target'],
                'feature_names': st.session_state.preprocessed_data['feature_names']
            }
    else:
        saved_models = [f.replace('.pkl', '') for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]
        if not saved_models:
            st.warning("No hay modelos guardados.")
        else:
            selected_model = st.selectbox("Selecciona un modelo guardado", saved_models)
            if st.button("Cargar Modelo"):
                model, preprocessor, model_info = load_model(selected_model)
                if model is not None:
                    st.success(f"Modelo '{selected_model}' cargado correctamente.")
    
    if model is not None and model_info is not None:
        st.subheader("Información del modelo")
        st.write(f"**Modelo:** {model_info['model_type']}")
        st.write(f"**Tipo:** {model_info['problem_type']}")
        
        prediction_option = st.radio("¿Cómo ingresar datos?", ["Manual", "Archivo CSV"])
        
        if prediction_option == "Manual":
            input_data = {}
            for feat in model_info['feature_names']:
                input_data[feat] = st.text_input(f"{feat}")
            
            if st.button("Predecir") and all(input_data.values()):
                try:
                    input_df = pd.DataFrame([input_data])
                    for col in input_df.columns:
                        try:
                            input_df[col] = pd.to_numeric(input_df[col])
                        except:
                            pass
                    if preprocessor and hasattr(preprocessor, 'transform'):
                        X_new = preprocessor.transform(input_df)
                    else:
                        X_new = input_df.values
                    pred = model.predict(X_new)
                    proba = None
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X_new)[0]
                    
                    st.subheader("Resultado")
                    if model_info['problem_type'] == "Clasificación":
                        st.success(f"Clase predicha: **{pred[0]}**")
                        if proba is not None:
                            proba_df = pd.DataFrame({'Clase': model.classes_, 'Probabilidad': proba})
                            st.dataframe(proba_df)
                            fig, ax = plt.subplots()
                            sns.barplot(x='Clase', y='Probabilidad', data=proba_df, ax=ax)
                            st.pyplot(fig)
                    else:
                        st.success(f"Valor predicho: **{pred[0]:.4f}**")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            uploaded_file = st.file_uploader("Sube CSV con datos para predecir", type=["csv"])
            if uploaded_file:
                try:
                    input_df = pd.read_csv(uploaded_file)
                    st.dataframe(input_df.head())
                    missing = [f for f in model_info['feature_names'] if f not in input_df.columns]
                    if missing:
                        st.error(f"Faltan columnas: {missing}")
                    elif st.button("Predecir todo"):
                        input_df = input_df[model_info['feature_names']]
                        if preprocessor and hasattr(preprocessor, 'transform'):
                            X_new = preprocessor.transform(input_df)
                        else:
                            X_new = input_df.values
                        preds = model.predict(X_new)
                        result = input_df.copy()
                        result['Predicción'] = preds
                        st.dataframe(result)
                        st.download_button(
                            label="⬇️ Descargar predicciones (CSV)",
                            data=result.to_csv(index=False),
                            file_name="predicciones.csv",
                            mime="text/csv"
                        )
                        
                        # Gráfico de resultados
                        if model_info['problem_type'] == "Regresión":
                            fig, ax = plt.subplots()
                            sns.histplot(preds, kde=True, ax=ax)
                            st.pyplot(fig)
                        else:
                            fig, ax = plt.subplots()
                            sns.countplot(x=preds, ax=ax)
                            st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error al procesar: {e}")