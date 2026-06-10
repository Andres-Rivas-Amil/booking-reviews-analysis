#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BOOKING REVIEWS ANALYSIS PIPELINE - VERSIÓN AUTOCONTENIDA
Ejecuta este archivo directamente y hace TODO el análisis.

Uso:
    python booking_analysis.py

No necesita archivos adicionales (solo el CSV en la ruta especificada)
"""

import os
import warnings
import pickle
import json
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error, r2_score
from textblob import TextBlob
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================
# CONFIGURACIÓN (RUTA YA CONFIGURADA)
# ============================================
CONFIG = {
    # Ruta completa a tu archivo CSV
    'data_path': r'C:\Users\andy_\Documents\Datasets\Hotel Review Booking\data\booking_reviews.csv',
    
    # Directorios de salida
    'output_dir': 'outputs/',
    'figures_dir': 'outputs/figures/',
    'models_dir': 'outputs/models/',
    'reports_dir': 'outputs/reports/',
    
    # Parámetros del modelo
    'test_size': 0.2,
    'random_state': 42,
    'max_features': 3000,
    'min_df': 10,
    'max_df': 0.8,
    'ridge_alpha': 1.0,
    
    # Clustering
    'min_reviews_per_hotel': 100,
    'n_clusters': 4,
    
    # Análisis de sentimiento
    'sentiment_sample_size': 30000,
    
    # Visualización
    'figure_dpi': 150,
}

# Crear directorios de salida
for dir_path in [CONFIG['output_dir'], CONFIG['figures_dir'], 
                 CONFIG['models_dir'], CONFIG['reports_dir']]:
    os.makedirs(dir_path, exist_ok=True)

# Configurar estilo
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = CONFIG['figure_dpi']
plt.rcParams['figure.figsize'] = (12, 6)


def load_data():
    """Cargar datos desde CSV"""
    print("\n" + "="*80)
    print("📂 CARGANDO DATOS")
    print("="*80)
    
    print(f"Buscando archivo en: {CONFIG['data_path']}")
    
    if not os.path.exists(CONFIG['data_path']):
        print(f"\n❌ ERROR: No se encuentra el archivo")
        print(f"   Ruta buscada: {CONFIG['data_path']}")
        print("\n📋 Soluciones:")
        print("   1. Verifica que la ruta sea correcta")
        print("   2. Asegúrate de que el archivo existe")
        print("   3. Modifica la ruta en CONFIG['data_path']")
        return None
    
    df = pd.read_csv(CONFIG['data_path'])
    print(f"\n✅ Datos cargados correctamente")
    print(f"   • Filas: {df.shape[0]:,}")
    print(f"   • Columnas: {df.shape[1]}")
    return df


def exploratory_analysis(df):
    """Análisis exploratorio inicial"""
    print("\n" + "="*80)
    print("📊 ANÁLISIS EXPLORATORIO INICIAL")
    print("="*80)
    
    print(f"\n🔍 Dimensiones: {df.shape[0]:,} filas, {df.shape[1]} columnas")
    print(f"\n❌ Valores nulos totales: {df.isnull().sum().sum():,}")
    print(f"\n🔄 Filas duplicadas: {df.duplicated().sum():,}")
    
    print("\n📊 Estadísticas de Reviewer_Score:")
    print(f"  • Media: {df['Reviewer_Score'].mean():.2f}")
    print(f"  • Mediana: {df['Reviewer_Score'].median():.2f}")
    print(f"  • Moda: {df['Reviewer_Score'].mode()[0]:.2f}")
    print(f"  • Desviación estándar: {df['Reviewer_Score'].std():.2f}")
    print(f"  • Mínimo: {df['Reviewer_Score'].min():.1f}")
    print(f"  • Máximo: {df['Reviewer_Score'].max():.1f}")
    
    return df


def run_etl(df):
    """Ejecutar ETL completo"""
    print("\n" + "="*80)
    print("🔄 EJECUTANDO ETL")
    print("="*80)
    
    # 1. Eliminar duplicados
    print("\n1️⃣ Eliminando duplicados...")
    initial_rows = len(df)
    df = df.drop_duplicates()
    print(f"   Eliminados: {initial_rows - len(df):,} duplicados")
    
    # 2. Limpiar coordenadas
    print("\n2️⃣ Limpiando coordenadas...")
    df = df.dropna(subset=['lat', 'lng'])
    print(f"   Eliminados: {initial_rows - len(df):,} filas sin coordenadas")
    
    # 3. Convertir tipos
    print("\n3️⃣ Convirtiendo tipos de datos...")
    df['Review_Date'] = pd.to_datetime(df['Review_Date'])
    df['days_since_review'] = pd.to_numeric(df['days_since_review'], errors='coerce')
    print(f"   ✓ Fechas convertidas")
    
    # 4. Limpiar texto
    print("\n4️⃣ Limpiando reseñas...")
    df['Positive_Review_Clean'] = df['Positive_Review'].replace('Sin positiva', '')
    df['Negative_Review_Clean'] = df['Negative_Review'].replace('Sin negativa', '')
    df['Full_Review'] = df['Positive_Review_Clean'] + ' ' + df['Negative_Review_Clean']
    df['Full_Review'] = df['Full_Review'].str.strip()
    print(f"   ✓ Texto limpiado")
    
    # 5. Extraer país y ciudad
    print("\n5️⃣ Extrayendo país y ciudad...")
    
    def extraer_pais(direccion):
        if pd.isna(direccion):
            return 'Desconocido'
        paises = {
            'Spain': 'España', 'France': 'Francia', 'Italy': 'Italia',
            'United Kingdom': 'Reino Unido', 'Netherlands': 'Países Bajos',
            'Austria': 'Austria', 'Germany': 'Alemania', 'Switzerland': 'Suiza'
        }
        for key, value in paises.items():
            if key.lower() in str(direccion).lower():
                return value
        return 'Otro'
    
    def extraer_ciudad(direccion):
        if pd.isna(direccion):
            return 'Desconocido'
        ciudades = {
            'London': 'Londres', 'Barcelona': 'Barcelona', 'Paris': 'París',
            'Amsterdam': 'Ámsterdam', 'Milan': 'Milán', 'Vienna': 'Viena',
            'Rome': 'Roma', 'Venice': 'Venecia', 'Florence': 'Florencia'
        }
        for key, value in ciudades.items():
            if key.lower() in str(direccion).lower():
                return value
        return 'Otra ciudad'
    
    df['Hotel_Country'] = df['Hotel_Address'].apply(extraer_pais)
    df['Hotel_City'] = df['Hotel_Address'].apply(extraer_ciudad)
    print(f"   ✓ Países extraídos: {df['Hotel_Country'].nunique()}")
    
    # 6. Crear variables derivadas
    print("\n6️⃣ Creando variables derivadas...")
    df['Review_Year'] = df['Review_Date'].dt.year
    df['Review_Month'] = df['Review_Date'].dt.month
    df['Review_Length_Chars'] = df['Full_Review'].str.len()
    
    def score_category(score):
        if score >= 9: return 'Excelente'
        elif score >= 8: return 'Muy Bueno'
        elif score >= 6: return 'Bueno'
        elif score >= 4: return 'Regular'
        else: return 'Malo'
    
    df['Score_Category'] = df['Reviewer_Score'].apply(score_category)
    print(f"   ✓ Variables creadas")
    
    print(f"\n✅ ETL completado. Dataset final: {len(df):,} filas")
    
    # Guardar checkpoint
    df.to_parquet(f"{CONFIG['output_dir']}/data_clean.parquet", index=False)
    print(f"💾 Dataset guardado en {CONFIG['output_dir']}/data_clean.parquet")
    
    return df


def generate_eda_visualizations(df):
    """Generar visualizaciones del EDA"""
    print("\n" + "="*80)
    print("📊 GENERANDO VISUALIZACIONES EDA")
    print("="*80)
    
    # 1. Distribución de puntuaciones
    print("1️⃣ Distribución de puntuaciones...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(df['Reviewer_Score'], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].set_title('Distribución de Puntuaciones')
    axes[0].set_xlabel('Reviewer Score')
    axes[0].axvline(df['Reviewer_Score'].mean(), color='red', linestyle='--', label=f'Media: {df["Reviewer_Score"].mean():.2f}')
    axes[0].legend()
    
    sns.boxplot(y=df['Reviewer_Score'], ax=axes[1], color='lightblue')
    axes[1].set_title('Boxplot')
    axes[1].set_ylabel('Reviewer Score')
    
    sns.kdeplot(df['Reviewer_Score'], ax=axes[2], fill=True, color='green')
    axes[2].set_title('Densidad')
    axes[2].set_xlabel('Reviewer Score')
    axes[2].axvline(df['Reviewer_Score'].median(), color='red', linestyle='--', label=f'Mediana: {df["Reviewer_Score"].median():.2f}')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(f"{CONFIG['figures_dir']}/1_score_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✓ Guardado: 1_score_distribution.png")
    
    # 2. Puntuación por país
    print("2️⃣ Puntuación por país...")
    pais_stats = df.groupby('Hotel_Country')['Reviewer_Score'].mean().sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    colors = ['gold' if i == 0 else 'silver' if i == 1 else 'coral' for i in range(len(pais_stats))]
    pais_stats.plot(kind='bar', color=colors, edgecolor='black')
    plt.title('Puntuación Media por País', fontweight='bold', fontsize=14)
    plt.xlabel('País')
    plt.ylabel('Reviewer Score')
    plt.xticks(rotation=45, ha='right')
    plt.axhline(y=df['Reviewer_Score'].mean(), color='red', linestyle='--', label=f'Media global: {df["Reviewer_Score"].mean():.2f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{CONFIG['figures_dir']}/2_score_by_country.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✓ Guardado: 2_score_by_country.png")
    
    # 3. Matriz de correlación
    print("3️⃣ Matriz de correlación...")
    numeric_cols = ['Reviewer_Score', 'Review_Total_Positive_Word_Counts', 
                    'Review_Total_Negative_Word_Counts', 'Total_Number_of_Reviews']
    corr_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, fmt='.3f', square=True)
    plt.title('Matriz de Correlación', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{CONFIG['figures_dir']}/3_correlation_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✓ Guardado: 3_correlation_matrix.png")
    
    # 4. Top mejores hoteles
    print("4️⃣ Top mejores hoteles...")
    hotel_stats = df.groupby('Hotel_Name')['Reviewer_Score'].mean().sort_values(ascending=False).head(10)
    
    plt.figure(figsize=(12, 6))
    hotel_stats.plot(kind='barh', color='green', alpha=0.7, edgecolor='black')
    plt.title('Top 10 Mejores Hoteles', fontweight='bold', fontsize=14)
    plt.xlabel('Puntuación Media')
    plt.tight_layout()
    plt.savefig(f"{CONFIG['figures_dir']}/4_top_hotels.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✓ Guardado: 4_top_hotels.png")
    
    print(f"\n✅ Todas las visualizaciones guardadas en {CONFIG['figures_dir']}")


def run_regression_model(df):
    """Modelo de regresión para predecir puntuaciones"""
    print("\n" + "="*80)
    print("🧠 MODELO DE REGRESIÓN: Palabras → Puntuación")
    print("="*80)
    
    X = df['Full_Review'].fillna('')
    y = df['Reviewer_Score']
    
    # División estratificada
    y_cat = pd.cut(y, bins=[0, 5, 7, 8, 9, 11], 
                   labels=['muy_bajo', 'bajo', 'medio', 'alto', 'muy_alto'])
    sss = StratifiedShuffleSplit(n_splits=1, test_size=CONFIG['test_size'], 
                                 random_state=CONFIG['random_state'])
    train_idx, test_idx = next(sss.split(X, y_cat))
    
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    
    print(f"\n📊 División de datos:")
    print(f"   • Train: {len(X_train):,} reseñas")
    print(f"   • Test: {len(X_test):,} reseñas")
    
    # Vectorizar
    print("\n🔄 Vectorizando texto con TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=CONFIG['max_features'],
        min_df=CONFIG['min_df'],
        max_df=CONFIG['max_df'],
        stop_words='english',
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"   • Vector creado: {X_train_tfidf.shape[1]} características")
    
    # Entrenar modelo
    print("\n🧠 Entrenando modelo Ridge Regression...")
    model = Ridge(alpha=CONFIG['ridge_alpha'], random_state=CONFIG['random_state'])
    model.fit(X_train_tfidf, y_train)
    
    # Evaluar
    y_pred = model.predict(X_test_tfidf)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"\n📈 Métricas del modelo:")
    print(f"   • R² Score: {r2:.4f}")
    print(f"   • RMSE: {rmse:.3f}")
    
    # Palabras más influyentes
    feature_names = vectorizer.get_feature_names_out()
    coef_df = pd.DataFrame({'palabra': feature_names, 'coeficiente': model.coef_})
    palabras_pos = coef_df.nlargest(15, 'coeficiente')
    palabras_neg = coef_df.nsmallest(15, 'coeficiente')
    
    print("\n🟢 TOP 10 PALABRAS QUE PREDICEN PUNTUACIÓN ALTA:")
    for _, row in palabras_pos.head(10).iterrows():
        print(f"   +{row['coeficiente']:.4f} → '{row['palabra']}'")
    
    print("\n🔴 TOP 10 PALABRAS QUE PREDICEN PUNTUACIÓN BAJA:")
    for _, row in palabras_neg.head(10).iterrows():
        print(f"   {row['coeficiente']:.4f} → '{row['palabra']}'")
    
    # Guardar modelos
    with open(f"{CONFIG['models_dir']}/regression_model.pkl", 'wb') as f:
        pickle.dump(model, f)
    with open(f"{CONFIG['models_dir']}/tfidf_vectorizer.pkl", 'wb') as f:
        pickle.dump(vectorizer, f)
    
    palabras_pos.to_csv(f"{CONFIG['reports_dir']}/palabras_positivas.csv", index=False)
    palabras_neg.to_csv(f"{CONFIG['reports_dir']}/palabras_negativas.csv", index=False)
    
    print(f"\n💾 Modelos guardados en {CONFIG['models_dir']}")
    
    return model, r2, rmse


def run_sentiment_analysis(df):
    """Análisis de sentimiento con TextBlob"""
    print("\n" + "="*80)
    print("💬 ANÁLISIS DE SENTIMIENTO")
    print("="*80)
    
    sample_size = min(CONFIG['sentiment_sample_size'], len(df))
    muestra = df.sample(n=sample_size, random_state=CONFIG['random_state'])
    
    def get_sentiment(text):
        if pd.isna(text) or len(str(text)) < 10:
            return 0
        try:
            blob = TextBlob(str(text))
            return blob.sentiment.polarity
        except:
            return 0
    
    print(f"📊 Analizando sentimiento de {sample_size:,} reseñas...")
    polaridades = []
    for text in tqdm(muestra['Full_Review'].values, desc="Procesando"):
        polaridades.append(get_sentiment(text))
    
    muestra['sentimiento'] = polaridades
    correlation = muestra['sentimiento'].corr(muestra['Reviewer_Score'])
    
    print(f"\n📈 Correlación sentimiento vs puntuación real: {correlation:.3f}")
    print(f"   • El sentimiento explica {correlation**2:.1%} de la variación")
    
    # Visualización
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(muestra['sentimiento'], muestra['Reviewer_Score'], alpha=0.2, s=1, c='steelblue')
    plt.xlabel('Sentimiento (TextBlob)')
    plt.ylabel('Reviewer Score')
    plt.title(f'Correlación: {correlation:.3f}')
    
    plt.subplot(1, 2, 2)
    sentimiento_por_categoria = muestra.groupby('Score_Category')['sentimiento'].mean().sort_values()
    sentimiento_por_categoria.plot(kind='bar', color='coral', edgecolor='black')
    plt.title('Sentimiento por Categoría de Puntuación')
    plt.xlabel('Categoría')
    plt.ylabel('Sentimiento Medio')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{CONFIG['figures_dir']}/sentiment_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Guardar resultados
    results = {
        'correlation': correlation,
        'sample_size': sample_size,
        'mean_sentiment': float(muestra['sentimiento'].mean()),
        'std_sentiment': float(muestra['sentimiento'].std())
    }
    
    with open(f"{CONFIG['reports_dir']}/sentiment_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Resultados guardados en {CONFIG['reports_dir']}")
    
    return correlation


def run_clustering(df):
    """Clustering de hoteles basado en opiniones"""
    print("\n" + "="*80)
    print("🏨 CLUSTERING DE HOTELES")
    print("="*80)
    
    # Agregar reseñas por hotel
    hotel_reviews = df.groupby('Hotel_Name').agg({
        'Full_Review': lambda x: ' '.join(x),
        'Reviewer_Score': 'mean',
        'Hotel_Country': 'first',
        'Hotel_City': 'first',
        'Total_Number_of_Reviews': 'first'
    }).reset_index()
    
    hotel_reviews = hotel_reviews[hotel_reviews['Total_Number_of_Reviews'] >= CONFIG['min_reviews_per_hotel']]
    print(f"📊 Hoteles analizados: {len(hotel_reviews)} (mínimo {CONFIG['min_reviews_per_hotel']} reseñas)")
    
    # Vectorizar
    print("\n🔄 Vectorizando reseñas de hoteles...")
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english', min_df=2)
    X_hotels = vectorizer.fit_transform(hotel_reviews['Full_Review'])
    print(f"   • Vector creado: {X_hotels.shape[1]} características")
    
    # Clustering
    print(f"\n📊 Ejecutando K-Means con {CONFIG['n_clusters']} clusters...")
    kmeans = KMeans(n_clusters=CONFIG['n_clusters'], random_state=CONFIG['random_state'], n_init=10)
    hotel_reviews['Cluster'] = kmeans.fit_predict(X_hotels)
    
    # Resumen por cluster
    cluster_summary = hotel_reviews.groupby('Cluster').agg({
        'Reviewer_Score': 'mean',
        'Total_Number_of_Reviews': 'mean',
        'Hotel_Country': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'mixed'
    }).round(2)
    
    cluster_summary.columns = ['Puntuación_Media', 'Reseñas_Promedio', 'País_Predominante']
    
    print("\n📊 CARACTERÍSTICAS DE CADA CLUSTER:")
    for cluster in range(CONFIG['n_clusters']):
        print(f"\n🔹 Cluster {cluster}:")
        print(f"   • Puntuación media: {cluster_summary.loc[cluster, 'Puntuación_Media']:.2f}")
        print(f"   • Reseñas promedio: {cluster_summary.loc[cluster, 'Reseñas_Promedio']:.0f}")
        print(f"   • País predominante: {cluster_summary.loc[cluster, 'País_Predominante']}")
    
    # Visualizar con PCA
    print("\n🎨 Visualizando clusters...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_hotels.toarray())
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=hotel_reviews['Cluster'], 
                          cmap='viridis', alpha=0.6, s=50)
    plt.colorbar(scatter, label='Cluster')
    plt.title('Visualización de Clusters de Hoteles (PCA)', fontweight='bold', fontsize=14)
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.tight_layout()
    plt.savefig(f"{CONFIG['figures_dir']}/clusters_pca.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Guardar resultados
    cluster_summary.to_csv(f"{CONFIG['reports_dir']}/cluster_summary.csv")
    with open(f"{CONFIG['models_dir']}/kmeans_model.pkl", 'wb') as f:
        pickle.dump(kmeans, f)
    with open(f"{CONFIG['models_dir']}/cluster_vectorizer.pkl", 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print(f"\n💾 Resultados guardados en {CONFIG['reports_dir']}")
    print(f"💾 Modelo guardado en {CONFIG['models_dir']}")
    
    return hotel_reviews, cluster_summary


def run_recommendation_engine(df):
    """Motor de recomendación de hoteles"""
    print("\n" + "="*80)
    print("🎯 MOTOR DE RECOMENDACIÓN DE HOTELES")
    print("="*80)
    
    def recomendar_hoteles(caracteristica, top_n=10):
        """Recomienda hoteles basados en una característica especial"""
        
        # Buscar hoteles que mencionan la característica
        hoteles_con_caracteristica = df[
            df['Positive_Review_Clean'].str.contains(caracteristica, case=False, na=False)
        ]['Hotel_Name'].unique()
        
        if len(hoteles_con_caracteristica) == 0:
            return None
        
        # Obtener estadísticas
        hoteles_recomendados = df[df['Hotel_Name'].isin(hoteles_con_caracteristica)].groupby('Hotel_Name').agg({
            'Reviewer_Score': 'mean',
            'Hotel_City': 'first',
            'Hotel_Country': 'first'
        }).round(2)
        
        n_resenas = df[df['Hotel_Name'].isin(hoteles_con_caracteristica)].groupby('Hotel_Name').size()
        hoteles_recomendados['N_Reseñas'] = n_resenas
        hoteles_recomendados = hoteles_recomendados[hoteles_recomendados['N_Reseñas'] >= 50]
        hoteles_recomendados = hoteles_recomendados.sort_values('Reviewer_Score', ascending=False)
        
        return hoteles_recomendados.head(top_n)
    
    # Ejemplos
    print("\n🔍 EJEMPLO 1: Hoteles con 'piscina'")
    resultados = recomendar_hoteles('pool', top_n=10)
    if resultados is not None:
        for idx, (hotel, row) in enumerate(resultados.iterrows(), 1):
            print(f"   {idx}. {hotel[:45]}... ⭐ {row['Reviewer_Score']} | {row['Hotel_City']} | {row['N_Reseñas']:.0f} reseñas")
    
    print("\n🔍 EJEMPLO 2: Hoteles con 'spa'")
    resultados = recomendar_hoteles('spa', top_n=10)
    if resultados is not None:
        for idx, (hotel, row) in enumerate(resultados.iterrows(), 1):
            print(f"   {idx}. {hotel[:45]}... ⭐ {row['Reviewer_Score']} | {row['Hotel_City']} | {row['N_Reseñas']:.0f} reseñas")
    
    print("\n🔍 EJEMPLO 3: Hoteles con 'desayuno'")
    resultados = recomendar_hoteles('breakfast', top_n=10)
    if resultados is not None:
        for idx, (hotel, row) in enumerate(resultados.iterrows(), 1):
            print(f"   {idx}. {hotel[:45]}... ⭐ {row['Reviewer_Score']} | {row['Hotel_City']} | {row['N_Reseñas']:.0f} reseñas")
    
    return recomendar_hoteles


def generate_final_report(df):
    """Generar reporte final resumen"""
    print("\n" + "="*80)
    print("📄 GENERANDO REPORTE FINAL")
    print("="*80)
    
    # Calcular estadísticas
    mejor_hotel = df.groupby('Hotel_Name')['Reviewer_Score'].mean().idxmax()
    mejor_puntuacion = df.groupby('Hotel_Name')['Reviewer_Score'].mean().max()
    
    peor_hotel = df.groupby('Hotel_Name')['Reviewer_Score'].mean().idxmin()
    peor_puntuacion = df.groupby('Hotel_Name')['Reviewer_Score'].mean().min()
    
    pais_stats = df.groupby('Hotel_Country')['Reviewer_Score'].mean().sort_values(ascending=False)
    
    report = {
        'fecha_analisis': datetime.now().isoformat(),
        'dataset': {
            'total_resenas': len(df),
            'total_hoteles': df['Hotel_Name'].nunique(),
            'total_paises': df['Hotel_Country'].nunique(),
            'total_ciudades': df['Hotel_City'].nunique(),
            'media_puntuacion': float(df['Reviewer_Score'].mean()),
            'mediana_puntuacion': float(df['Reviewer_Score'].median()),
            'desviacion_puntuacion': float(df['Reviewer_Score'].std())
        },
        'top_paises': {k: float(v) for k, v in pais_stats.head(3).items()},
        'bottom_paises': {k: float(v) for k, v in pais_stats.tail(3).items()},
        'mejor_hotel': {'nombre': mejor_hotel, 'puntuacion': float(mejor_puntuacion)},
        'peor_hotel': {'nombre': peor_hotel, 'puntuacion': float(peor_puntuacion)},
        'distribucion_categorias': df['Score_Category'].value_counts(normalize=True).mul(100).round(1).to_dict()
    }
    
    with open(f"{CONFIG['reports_dir']}/final_report.json", 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Mostrar resumen
    print("\n" + "="*80)
    print("📊 RESUMEN EJECUTIVO")
    print("="*80)
    
    print(f"\n📈 DATOS GENERALES:")
    print(f"   • Total reseñas: {report['dataset']['total_resenas']:,}")
    print(f"   • Total hoteles: {report['dataset']['total_hoteles']:,}")
    print(f"   • Total países: {report['dataset']['total_paises']}")
    print(f"   • Puntuación media global: {report['dataset']['media_puntuacion']:.2f}")
    
    print(f"\n🏆 TOP PAÍSES:")
    for pais, score in report['top_paises'].items():
        print(f"   • {pais}: {score:.2f}")
    
    print(f"\n🏨 MEJOR HOTEL:")
    print(f"   • {report['mejor_hotel']['nombre'][:60]} ({report['mejor_hotel']['puntuacion']:.2f})")
    
    print(f"\n📊 DISTRIBUCIÓN DE PUNTUACIONES:")
    for cat, pct in report['distribucion_categorias'].items():
        print(f"   • {cat}: {pct:.1f}%")
    
    print(f"\n💾 Reporte guardado en {CONFIG['reports_dir']}/final_report.json")


def run_full_pipeline():
    """Ejecutar pipeline completo"""
    print("="*80)
    print("🚀 PIPELINE DE ANÁLISIS DE REVIEWS DE BOOKING")
    print("="*80)
    print("\nEste pipeline realizará:")
    print("  1. Carga de datos")
    print("  2. ETL (limpieza y transformación)")
    print("  3. Análisis exploratorio (EDA)")
    print("  4. Visualizaciones")
    print("  5. Modelo de regresión (palabras → puntuación)")
    print("  6. Análisis de sentimiento")
    print("  7. Clustering de hoteles")
    print("  8. Motor de recomendación")
    print("  9. Reporte final")
    
    start_time = datetime.now()
    
    # Ejecutar cada paso
    df = load_data()
    if df is None:
        return
    
    df = exploratory_analysis(df)
    df = run_etl(df)
    generate_eda_visualizations(df)
    run_regression_model(df)
    run_sentiment_analysis(df)
    run_clustering(df)
    run_recommendation_engine(df)
    generate_final_report(df)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
    print(f"⏱️ Tiempo total: {duration:.2f} segundos")
    print(f"📁 Todos los resultados están en la carpeta 'outputs/'")
    print("="*80)


# ============================================
# EJECUCIÓN PRINCIPAL
# ============================================
if __name__ == "__main__":
    run_full_pipeline()

