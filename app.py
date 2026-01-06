from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import pandas as pd
from pathlib import Path
import json
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from anomaly_detection import AnomalyDetector, AnomalyVisualizer
from feature_extraction import FeatureExtractor
import traceback
import os

# Configuración de Flask
app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB max

# Variables globales para almacenar estado
detector = None
features = None
labels = None
paths = None
results = None
label_to_class = {0: 'Gato', 1: 'Perro', 2: 'Elefante', 3: 'Caballo', 4: 'León'}

def convert_to_native_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, dict):
        return {k: convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    return obj

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/animals/<path:filepath>')
def serve_animals(filepath):
    try:
        from flask import send_file
        full_path = Path('animals') / filepath
        
        if full_path.exists() and full_path.is_file():
            return send_file(str(full_path), mimetype='image/jpeg')
        else:
            # Retornar imagen placeholder si no existe
            return send_file('static/placeholder.png', mimetype='image/png') if Path('static/placeholder.png').exists() else '', 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/load-features', methods=['POST'])
def load_features_api():
    try:
        global features, labels, paths, detector
        
        # Opción 1: Archivo subido
        if 'file' in request.files:
            file = request.files['file']
            
            if file.filename == '':
                return jsonify({
                    'success': False,
                    'error': 'No se seleccionó archivo'
                }), 400
            
            if not file.filename.endswith('.pkl'):
                return jsonify({
                    'success': False,
                    'error': 'El archivo debe ser .pkl'
                }), 400
            
            # Guardar temporalmente
            temp_path = Path('temp_features.pkl')
            file.save(str(temp_path))
            filepath = str(temp_path)
        
        # Opción 2: Ruta especificada
        else:
            data = request.json or {}
            filepath = data.get('filepath', 'features_train.pkl')
            
            if not Path(filepath).exists():
                return jsonify({
                    'success': False,
                    'error': f'Archivo no encontrado: {filepath}'
                }), 400
        
        # Cargar características
        features, labels, paths = FeatureExtractor.load_features(filepath)
        
        return jsonify({
            'success': True,
            'message': 'Características cargadas correctamente',
            'shape': convert_to_native_types(features.shape),
            'n_classes': int(len(np.unique(labels))),
            'unique_labels': convert_to_native_types(list(np.unique(labels)))
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/detect-anomalies', methods=['POST'])
def detect_anomalies_api():
    try:
        global detector, features, labels, paths, results
        
        if features is None:
            return jsonify({
                'success': False,
                'error': 'Primero carga las características'
            }), 400
        
        data = request.json
        contamination = float(data.get('contamination', 0.05))
        n_estimators = int(data.get('n_estimators', 100))
        
        # Crear detector
        detector = AnomalyDetector(
            features=features,
            labels=labels,
            contamination=contamination
        )
        
        # Ejecutar detección
        labels_pred, scores = detector.detect_isolation_forest(
            n_estimators=n_estimators,
            random_state=42
        )
        
        # Convertir a tipos nativos Python
        labels_pred = labels_pred.astype(int)
        scores = scores.astype(float)
        
        # Guardar resultados
        results = {
            'labels_pred': labels_pred,
            'scores': scores,
            'n_anomalies': int(np.sum(labels_pred == -1)),
            'n_normal': int(np.sum(labels_pred == 1)),
            'contamination': float(contamination)
        }
        
        # Calcular estadísticas
        n_total = int(len(labels_pred))
        n_anomalies = int(results['n_anomalies'])
        n_normal = int(results['n_normal'])
        pct_anomalies = float(round(n_anomalies / n_total * 100, 2))
        
        scores_normal = scores[labels_pred == 1]
        scores_anomaly = scores[labels_pred == -1]
        
        score_mean_normal = float(round(scores_normal.mean(), 4)) if len(scores_normal) > 0 else 0.0
        score_mean_anomaly = float(round(scores_anomaly.mean(), 4)) if len(scores_anomaly) > 0 else 0.0
        score_min = float(round(scores.min(), 4))
        score_max = float(round(scores.max(), 4))
        
        return jsonify({
            'success': True,
            'message': 'Detección completada',
            'results': {
                'n_anomalies': n_anomalies,
                'n_normal': n_normal,
                'contamination': float(contamination)
            },
            'stats': {
                'n_total': n_total,
                'n_anomalies': n_anomalies,
                'n_normal': n_normal,
                'pct_anomalies': pct_anomalies,
                'score_mean_normal': score_mean_normal,
                'score_mean_anomaly': score_mean_anomaly,
                'score_min': score_min,
                'score_max': score_max
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/analyze-anomalies', methods=['POST'])
def analyze_anomalies_api():
    try:
        if detector is None or results is None:
            return jsonify({
                'success': False,
                'error': 'Primero ejecuta la detección'
            }), 400
        
        data = request.json
        top_n = int(data.get('top_n', 20))
        
        # Análisis
        anomalies, scores = detector.analyze_anomalies(top_n=top_n)
        
        # Crear tabla de anomalías
        anomaly_data = []
        for rank, (idx, score) in enumerate(zip(anomalies, scores), 1):
            anomaly_data.append({
                'rank': int(rank),
                'index': int(idx),
                'score': float(round(score, 6)),
                'class': int(labels[idx]),
                'class_name': label_to_class.get(int(labels[idx]), f'Clase {labels[idx]}'),
                'path': str(paths[idx]) if paths is not None else ''
            })
        
        # Análisis por clase
        df_by_class = detector.compare_anomalies_by_class()
        by_class_data = []
        if df_by_class is not None:
            for _, row in df_by_class.iterrows():
                by_class_data.append({
                    'class': str(row['Clase']),
                    'total': int(row['Total']),
                    'anomalies': int(row['Anomalías']),
                    'pct_anomalies': str(row['% Anomalías']),
                    'score_mean': str(row['Score Medio']),
                    'score_min': str(row['Score Mín']),
                    'score_max': str(row['Score Máx'])
                })
        
        return jsonify({
            'success': True,
            'anomalies': anomaly_data,
            'by_class': by_class_data
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/compare-contaminations', methods=['POST'])
def compare_contaminations_api():
    try:
        global detector, features, labels
        
        if features is None:
            return jsonify({
                'success': False,
                'error': 'Primero carga las características'
            }), 400
        
        data = request.json
        contaminations = [float(x) for x in data.get('contaminations', [0.01, 0.05, 0.10, 0.15, 0.20])]
        
        detector = AnomalyDetector(features=features, labels=labels, contamination=0.05)
        results_multi = detector.detect_with_multiple_contaminations(contaminations=contaminations)
        
        comparison_data = []
        for cont, result in sorted(results_multi.items()):
            n_anom = int(result['n_anomalies'])
            comparison_data.append({
                'contamination': f'{float(cont)*100:.1f}%',
                'n_anomalies': n_anom,
                'pct_total': float(round(n_anom / len(labels) * 100, 2)),
                'score_mean': float(round(result['scores'][result['labels'] == -1].mean(), 4)) if np.sum(result['labels'] == -1) > 0 else 0.0
            })
        
        return jsonify({
            'success': True,
            'comparison': comparison_data
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/visualize-anomalies', methods=['POST'])
def visualize_anomalies_api():
    try:
        if detector is None or results is None:
            return jsonify({
                'success': False,
                'error': 'Primero ejecuta la detección'
            }), 400
        
        # Reducir a 2D
        features_2d = detector.reduce_to_2d_pca()
        
        # Crear visualizador
        visualizer = AnomalyVisualizer(
            features_2d=features_2d,
            label_to_class=label_to_class
        )
        
        # Generar gráfico principal
        fig, ax = visualizer.plot_anomalies(
            results['labels_pred'],
            results['scores'],
            title='Detección de Anomalías - Isolation Forest'
        )
        
        # Convertir a base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close(fig)
        
        return jsonify({
            'success': True,
            'image': f'data:image/png;base64,{img_base64}'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/visualize-distribution', methods=['POST'])
def visualize_distribution_api():
    try:
        if detector is None or results is None:
            return jsonify({
                'success': False,
                'error': 'Primero ejecuta la detección'
            }), 400
        
        # Reducir a 2D
        features_2d = detector.reduce_to_2d_pca()
        
        # Crear visualizador
        visualizer = AnomalyVisualizer(features_2d=features_2d)
        
        # Generar gráfico
        fig, ax = visualizer.plot_score_distribution(
            results['scores'],
            results['labels_pred']
        )
        
        # Convertir a base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close(fig)
        
        return jsonify({
            'success': True,
            'image': f'data:image/png;base64,{img_base64}'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/visualize-by-class', methods=['POST'])
def visualize_by_class_api():
    try:
        if detector is None or results is None or labels is None:
            return jsonify({
                'success': False,
                'error': 'Primero ejecuta la detección'
            }), 400
        
        # Reducir a 2D
        features_2d = detector.reduce_to_2d_pca()
        
        # Crear visualizador
        visualizer = AnomalyVisualizer(
            features_2d=features_2d,
            label_to_class=label_to_class
        )
        
        # Generar gráfico
        fig, ax = visualizer.plot_by_class(
            labels,
            results['labels_pred'],
            results['scores'],
            label_to_class=label_to_class
        )
        
        # Convertir a base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close(fig)
        
        return jsonify({
            'success': True,
            'image': f'data:image/png;base64,{img_base64}'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/export-results', methods=['POST'])
def export_results_api():
    """Exporta resultados como CSV"""
    try:
        if detector is None or results is None:
            return jsonify({
                'success': False,
                'error': 'Primero ejecuta la detección'
            }), 400
        
        # Crear DataFrame
        df = pd.DataFrame({
            'index': range(len(results['labels_pred'])),
            'label': results['labels_pred'],
            'score': results['scores'],
            'class': labels,
            'class_name': [label_to_class.get(int(l), f'Clase {l}') for l in labels],
            'is_anomaly': results['labels_pred'] == -1
        })
        
        # Crear CSV en memoria
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        
        return jsonify({
            'success': True,
            'csv': csv_buffer.getvalue()
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/status', methods=['GET'])
def status_api():
    """Devuelve estado actual de la aplicación"""
    return jsonify({
        'features_loaded': features is not None,
        'features_shape': convert_to_native_types(features.shape) if features is not None else None,
        'detection_done': results is not None,
        'n_anomalies': results['n_anomalies'] if results else None
    })

@app.errorhandler(404)
def not_found(error):
    """Página no encontrada"""
    return jsonify({'error': 'Página no encontrada'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Error interno del servidor"""
    return jsonify({'error': 'Error interno del servidor'}), 500

if __name__ == '__main__':
    
    # Ejecutar app
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        threaded=True
    )