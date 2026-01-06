import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, efficientnet_b0
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class AnimalDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        self.label_to_idx = {}
        
        # Buscar todas las imágenes en subcarpetas
        idx = 0
        for class_dir in sorted(self.data_dir.iterdir()):
            if class_dir.is_dir():
                class_name = class_dir.name
                self.label_to_idx[class_name] = idx
                
                for img_path in class_dir.glob('*'):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                        self.images.append(str(img_path))
                        self.labels.append(idx)
                
                idx += 1
        
        print(f"Dataset cargado: {len(self.images)} imágenes, {len(self.label_to_idx)} clases")
        print(f"Clases: {self.label_to_idx}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        try:
            img = Image.open(self.images[idx]).convert('RGB')
            label = self.labels[idx]
            
            if self.transform:
                img = self.transform(img)
            
            return img, label, self.images[idx]
        except Exception as e:
            print(f"Error cargando {self.images[idx]}: {e}")
            return torch.zeros(3, 224, 224), -1, self.images[idx]

class FeatureExtractor:
    def __init__(self, model_name='resnet50', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model_name = model_name
        
        print(f"Usando device: {self.device}")
        
        # Cargar modelo preentrenado
        if model_name == 'resnet50':
            self.model = resnet50(pretrained=True)
            # Remover última capa (clasificación)
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            self.feature_dim = 2048
        elif model_name == 'efficientnet':
            self.model = efficientnet_b0(pretrained=True)
            # Remover última capa
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            self.feature_dim = 1280
        
        self.model = self.model.to(device)
        self.model.eval()
    
    def extract_features(self, data_loader):
        """Extrae características de todas las imágenes"""
        features_list = []
        labels_list = []
        paths_list = []
        
        with torch.no_grad():
            for images, labels, paths in tqdm(data_loader, desc="Extrayendo características"):
                images = images.to(self.device)
                
                # Forward pass
                feats = self.model(images)
                feats = feats.view(feats.size(0), -1)  # Flatten
                
                features_list.append(feats.cpu().numpy())
                labels_list.extend(labels.numpy())
                paths_list.extend(paths)
        
        # Concatenar todos los features
        features = np.vstack(features_list)
        labels = np.array(labels_list)
        
        return features, labels, paths_list
    
    def save_features(self, features, labels, paths, output_path='features.pkl'):
        """Guarda características en caché"""
        data = {
            'features': features,
            'labels': labels,
            'paths': paths,
            'model': self.model_name,
            'feature_dim': self.feature_dim
        }
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Características guardadas en {output_path}")
    
    @staticmethod
    def load_features(filepath='features.pkl'):
        """Carga características desde caché"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"Características cargadas desde {filepath}")
        return data['features'], data['labels'], data['paths']

def main():
    # Configuración
    DATA_DIR = 'animals'  # Cambia según tu estructura
    BATCH_SIZE = 32
    MODEL_NAME = 'resnet50'  # 'resnet50' o 'efficientnet'
    OUTPUT_FEATURES = 'features_train.pkl'
    
    # Transformaciones
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Cargar dataset
    print(f"\n{'='*60}")
    print(f"Cargando dataset desde: {DATA_DIR}")
    print(f"{'='*60}\n")
    dataset = AnimalDataset(DATA_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Extracción de características
    print(f"\n{'='*60}")
    print(f"Extrayendo características con {MODEL_NAME}")
    print(f"{'='*60}\n")
    extractor = FeatureExtractor(model_name=MODEL_NAME)
    features, labels, paths = extractor.extract_features(dataloader)
    
    # Guardar características
    print(f"\nDimensiones de características: {features.shape}")
    print(f"Número de clases: {len(np.unique(labels))}")
    extractor.save_features(features, labels, paths, OUTPUT_FEATURES)
    
    return features, labels, paths, dataset.label_to_idx

if __name__ == '__main__':
    main()