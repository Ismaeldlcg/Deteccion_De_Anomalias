import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
from pathlib import Path
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class AnomalyDetector:
    def __init__(self, features, labels=None, contamination=0.05):
        self.features_original = features
        self.labels_true = labels
        self.contamination = contamination
        
        # Normalizar características
        self.scaler = StandardScaler()
        self.features_scaled = self.scaler.fit_transform(features)
        
        print(f"\n{'='*70}")
        print(f"INICIALIZANDO DETECTOR DE ANOMALÍAS")
        print(f"{'='*70}\n")
        print(f"Número de muestras: {features.shape[0]}")
        print(f"Número de características: {features.shape[1]}")
        print(f"Tasa de contaminación esperada: {contamination*100:.1f}%")
        print(f"Anomalías esperadas: ~{int(features.shape[0] * contamination)}")
        
        self.results = {}
        self.iso_forest = None
    
    def detect_isolation_forest(self, n_estimators=100, max_samples='auto', 
                               max_features=1.0, random_state=42):
        print(f"\n{'='*70}")
        print(f"ISOLATION FOREST")
        print(f"{'='*70}\n")
        print(f"Entrenando modelo con {n_estimators} árboles...")
        
        self.iso_forest = IsolationForest(
            contamination=self.contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1
        )
        
        # Entrenar y predecir
        labels_pred = self.iso_forest.fit_predict(self.features_scaled)
        scores = self.iso_forest.score_samples(self.features_scaled)
        
        # -1 = anomalía, 1 = normal
        n_anomalies = np.sum(labels_pred == -1)
        n_normal = np.sum(labels_pred == 1)
        
        self.results['isolation_forest'] = {
            'labels': labels_pred,  # -1: anomalía, 1: normal
            'scores': scores,  # puntuación de anomalía (menor = más anómalo)
            'n_anomalies': n_anomalies,
            'n_normal': n_normal,
            'threshold': self.iso_forest.offset_,
            'n_estimators': n_estimators
        }
        
        print(f"Entrenamiento completado")
        print(f"\nResultados:")
        print(f"  - Muestras normales: {n_normal} ({n_normal/len(labels_pred)*100:.1f}%)")
        print(f"  - Anomalías detectadas: {n_anomalies} ({n_anomalies/len(labels_pred)*100:.1f}%)")
        print(f"  - Umbral de anomalía: {self.iso_forest.offset_:.4f}")
        print(f"  - Rango de puntuaciones: [{scores.min():.4f}, {scores.max():.4f}]")
        
        # Análisis estadístico
        self._print_statistics(labels_pred, scores)
        
        return labels_pred, scores
    
    def detect_with_multiple_contaminations(self, contaminations=[0.02, 0.05, 0.1, 0.15]):
        print(f"\n{'='*70}")
        print(f"PRUEBA CON MÚLTIPLES TASAS DE CONTAMINACIÓN")
        print(f"{'='*70}\n")
        
        results_multi = {}
        
        for cont in contaminations:
            print(f"Probando con contaminación = {cont*100:.1f}%...", end=" ")
            
            iso_forest = IsolationForest(
                contamination=cont,
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
            
            labels_pred = iso_forest.fit_predict(self.features_scaled)
            scores = iso_forest.score_samples(self.features_scaled)
            
            n_anomalies = np.sum(labels_pred == -1)
            
            results_multi[cont] = {
                'labels': labels_pred,
                'scores': scores,
                'n_anomalies': n_anomalies,
                'threshold': iso_forest.offset_
            }
            
            print(f"Anomalías: {n_anomalies}")
        
        self.results['multi_contamination'] = results_multi
        return results_multi
    
    def _print_statistics(self, labels_pred, scores):
        print(f"\nEstadísticas de Puntuaciones:")
        print(f"  - Media (normales): {scores[labels_pred == 1].mean():.4f}")
        print(f"  - Media (anomalías): {scores[labels_pred == -1].mean():.4f}")
        print(f"  - Desv. Est. (normales): {scores[labels_pred == 1].std():.4f}")
        print(f"  - Desv. Est. (anomalías): {scores[labels_pred == -1].std():.4f}")
        
        # Percentiles
        print(f"\nPercentiles de Puntuaciones:")
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            val = np.percentile(scores, p)
            print(f"  - P{p}: {val:.4f}")
    
    def analyze_anomalies(self, top_n=20):
        if 'isolation_forest' not in self.results:
            print("Primero ejecuta detect_isolation_forest()")
            return
        
        print(f"\n{'='*70}")
        print(f"ANÁLISIS DE ANOMALÍAS DETECTADAS (Top {top_n})")
        print(f"{'='*70}\n")
        
        result = self.results['isolation_forest']
        labels_pred = result['labels']
        scores = result['scores']
        
        # Obtener índices de anomalías
        anomaly_indices = np.where(labels_pred == -1)[0]
        anomaly_scores = scores[anomaly_indices]
        
        # Ordenar por puntuación (más anómalo primero)
        sorted_indices = np.argsort(anomaly_scores)[:top_n]
        
        print(f"Top {min(top_n, len(anomaly_indices))} anomalías más significativas:\n")
        print(f"{'Rank':<6} {'Índice':<8} {'Puntuación':<15} {'Clase Real':<15}")
        print("-" * 50)
        
        for rank, local_idx in enumerate(sorted_indices, 1):
            global_idx = anomaly_indices[local_idx]
            score = anomaly_scores[local_idx]
            class_label = self.labels_true[global_idx] if self.labels_true is not None else "?"
            
            print(f"{rank:<6} {global_idx:<8} {score:<15.6f} {class_label:<15}")
        
        return anomaly_indices, anomaly_scores
    
    def compare_anomalies_by_class(self):
        if 'isolation_forest' not in self.results or self.labels_true is None:
            print("Se requieren resultados de IF y etiquetas verdaderas")
            return
        
        print(f"\n{'='*70}")
        print(f"ANÁLISIS DE ANOMALÍAS POR CLASE")
        print(f"{'='*70}\n")
        
        result = self.results['isolation_forest']
        labels_pred = result['labels']
        scores = result['scores']
        
        unique_classes = np.unique(self.labels_true)
        
        comparison_data = []
        
        for class_idx in unique_classes:
            mask = self.labels_true == class_idx
            class_labels = labels_pred[mask]
            class_scores = scores[mask]
            
            n_total = np.sum(mask)
            n_anomalies = np.sum(class_labels == -1)
            pct_anomalies = (n_anomalies / n_total) * 100
            
            comparison_data.append({
                'Clase': f'Clase {class_idx}',
                'Total': n_total,
                'Anomalías': n_anomalies,
                '% Anomalías': f'{pct_anomalies:.2f}%',
                'Score Medio': f'{class_scores.mean():.4f}',
                'Score Mín': f'{class_scores.min():.4f}',
                'Score Máx': f'{class_scores.max():.4f}'
            })
        
        df = pd.DataFrame(comparison_data)
        print(df.to_string(index=False))
        
        return df
    
    def reduce_to_2d_pca(self, n_components=2):
        print(f"\n{'='*70}")
        print(f"Reduciendo a {n_components}D con PCA para visualización...")
        print(f"{'='*70}\n")
        
        pca = PCA(n_components=n_components)
        features_2d = pca.fit_transform(self.features_scaled)
        
        self.features_2d = features_2d
        self.pca = pca
        
        variance_explained = np.sum(pca.explained_variance_ratio_)
        print(f"Varianza explicada: {variance_explained:.4f}")
        print(f"Características 2D generadas: {features_2d.shape}")
        
        return features_2d
    
    def save_results(self, output_path='anomaly_results.pkl'):
        data = {
            'results': self.results,
            'features_scaled': self.features_scaled,
            'features_2d': getattr(self, 'features_2d', None),
            'labels_true': self.labels_true,
            'contamination': self.contamination,
            'iso_forest': self.iso_forest
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"\nResultados guardados en {output_path}")
    
    @staticmethod
    def load_results(filepath='anomaly_results.pkl'):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"Resultados cargados desde {filepath}")
        return data


class AnomalyVisualizer:
    def __init__(self, features_2d, label_to_class=None):
        self.features_2d = features_2d
        self.label_to_class = label_to_class or {}
    
    def plot_anomalies(self, labels_pred, scores, title="Anomaly Detection", figsize=(12, 8)):
        fig, ax = plt.subplots(figsize=figsize)
        
        # Separa normales de anomalías
        mask_normal = labels_pred == 1
        mask_anomaly = labels_pred == -1
        
        # Grafica normales
        scatter1 = ax.scatter(
            self.features_2d[mask_normal, 0],
            self.features_2d[mask_normal, 1],
            c=scores[mask_normal],
            cmap='Blues',
            s=60,
            alpha=0.6,
            label='Normal',
            edgecolors='blue',
            linewidth=0.5
        )
        
        # Grafica anomalías
        scatter2 = ax.scatter(
            self.features_2d[mask_anomaly, 0],
            self.features_2d[mask_anomaly, 1],
            c=scores[mask_anomaly],
            cmap='Reds',
            s=200,
            alpha=0.8,
            marker='X',
            label='Anomalía',
            edgecolors='darkred',
            linewidth=2
        )
        
        ax.set_xlabel('Componente 1', fontsize=12)
        ax.set_ylabel('Componente 2', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        
        # Colorbar
        cbar = plt.colorbar(scatter1, ax=ax, label='Puntuación de Anomalía')
        
        plt.tight_layout()
        return fig, ax
    
    def plot_score_distribution(self, scores, labels_pred, figsize=(12, 6)):
        fig, ax = plt.subplots(figsize=figsize)
        
        normal_scores = scores[labels_pred == 1]
        anomaly_scores = scores[labels_pred == -1]
        
        ax.hist(normal_scores, bins=50, alpha=0.6, label='Normal', color='blue', edgecolor='black')
        ax.hist(anomaly_scores, bins=30, alpha=0.6, label='Anomalía', color='red', edgecolor='black')
        
        # Línea del umbral
        threshold = self.features_2d if hasattr(self, 'threshold') else np.median(scores)
        ax.axvline(x=0, color='green', linestyle='--', linewidth=2, label='Umbral')
        
        ax.set_xlabel('Puntuación de Anomalía', fontsize=12)
        ax.set_ylabel('Frecuencia', fontsize=12)
        ax.set_title('Distribución de Puntuaciones de Anomalía', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig, ax
    
    def plot_by_class(self, labels_true, labels_pred, scores, label_to_class=None, figsize=(14, 8)):
        fig, ax = plt.subplots(figsize=figsize)
        
        unique_classes = np.unique(labels_true)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
        
        # Grafica cada clase
        for idx, class_label in enumerate(unique_classes):
            mask = labels_true == class_label
            
            # Normales de esta clase
            mask_normal = mask & (labels_pred == 1)
            if np.sum(mask_normal) > 0:
                class_name = label_to_class.get(class_label, f'Clase {class_label}')
                ax.scatter(
                    self.features_2d[mask_normal, 0],
                    self.features_2d[mask_normal, 1],
                    c=[colors[idx]],
                    s=60,
                    alpha=0.6,
                    label=f'{class_name} (Normal)',
                    edgecolors='black',
                    linewidth=0.3
                )
            
            # Anomalías de esta clase
            mask_anomaly = mask & (labels_pred == -1)
            if np.sum(mask_anomaly) > 0:
                class_name = label_to_class.get(class_label, f'Clase {class_label}')
                ax.scatter(
                    self.features_2d[mask_anomaly, 0],
                    self.features_2d[mask_anomaly, 1],
                    c=[colors[idx]],
                    s=200,
                    alpha=0.9,
                    marker='X',
                    label=f'{class_name} (Anomalía)',
                    edgecolors='black',
                    linewidth=1.5
                )
        
        ax.set_xlabel('Componente 1', fontsize=12)
        ax.set_ylabel('Componente 2', fontsize=12)
        ax.set_title('Detección de Anomalías por Clase', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_comparison_contaminations(self, results_multi, figsize=(16, 10)):
        contaminations = sorted(results_multi.keys())
        n_cols = 2
        n_rows = (len(contaminations) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()
        
        for idx, cont in enumerate(contaminations):
            ax = axes[idx]
            result = results_multi[cont]
            labels_pred = result['labels']
            scores = result['scores']
            
            mask_normal = labels_pred == 1
            mask_anomaly = labels_pred == -1
            
            ax.scatter(
                self.features_2d[mask_normal, 0],
                self.features_2d[mask_normal, 1],
                c='blue',
                s=40,
                alpha=0.5,
                label='Normal'
            )
            
            ax.scatter(
                self.features_2d[mask_anomaly, 0],
                self.features_2d[mask_anomaly, 1],
                c='red',
                s=150,
                alpha=0.8,
                marker='X',
                label='Anomalía'
            )
            
            ax.set_title(f'Contaminación = {cont*100:.1f}% (N={result["n_anomalies"]})',
                        fontweight='bold')
            ax.set_xlabel('Componente 1')
            ax.set_ylabel('Componente 2')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        # Elimina subplots vacíos
        for idx in range(len(contaminations), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.suptitle('Comparación de Diferentes Tasas de Contaminación', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        return fig, axes
    
    def save_figure(self, fig, output_path):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figura guardada: {output_path}")


def main():
    from feature_extraction import FeatureExtractor
    
    print("\n" + "="*70)
    print("PIPELINE DE DETECCIÓN DE ANOMALÍAS")
    print("="*70)
    
    # Cargar características
    print("\nCargando características...")
    features, labels, paths = FeatureExtractor.load_features('features_train.pkl')
    
    # Crear detector
    detector = AnomalyDetector(features, labels=labels, contamination=0.05)
    
    # Detección con Isolation Forest
    print("\n" + "="*70)
    print("EJECUCIÓN PRINCIPAL")
    print("="*70)
    labels_pred, scores = detector.detect_isolation_forest(n_estimators=100)
    
    # Analizar anomalías
    detector.analyze_anomalies(top_n=20)
    
    # Comparar por clase
    detector.compare_anomalies_by_class()
    
    # Probar múltiples contaminaciones
    results_multi = detector.detect_with_multiple_contaminations(
        contaminations=[0.02, 0.05, 0.1, 0.15, 0.2]
    )
    
    # Reducir a 2D
    features_2d = detector.reduce_to_2d_pca(n_components=2)
    
    # Guardar resultados
    detector.save_results('anomaly_results.pkl')
    
    # Visualizaciones
    print("\n" + "="*70)
    print("GENERANDO VISUALIZACIONES")
    print("="*70)
    
    label_to_class = {0: 'Gato', 1: 'Perro', 2: 'Elefante', 3: 'Caballo', 4: 'León'}
    visualizer = AnomalyVisualizer(features_2d, label_to_class=label_to_class)
    
    output_dir = Path('results/anomaly_detection')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualizar anomalías principales
    fig1, _ = visualizer.plot_anomalies(
        labels_pred, scores,
        title="Detección de Anomalías - Isolation Forest"
    )
    visualizer.save_figure(fig1, f'{output_dir}/anomalies_main.png')
    plt.close(fig1)
    
    # Distribución de puntuaciones
    fig2, _ = visualizer.plot_score_distribution(scores, labels_pred)
    visualizer.save_figure(fig2, f'{output_dir}/score_distribution.png')
    plt.close(fig2)
    
    # Por clase
    fig3, _ = visualizer.plot_by_class(labels, labels_pred, scores, label_to_class)
    visualizer.save_figure(fig3, f'{output_dir}/anomalies_by_class.png')
    plt.close(fig3)
    
    # Comparación de contaminaciones
    fig4, _ = visualizer.plot_comparison_contaminations(results_multi)
    visualizer.save_figure(fig4, f'{output_dir}/contaminations_comparison.png')
    plt.close(fig4)
    
    print(f"\n✓ Visualizaciones guardadas en: {output_dir}\n")
    
    return detector, visualizer, results_multi


if __name__ == '__main__':
    detector, visualizer, results_multi = main()