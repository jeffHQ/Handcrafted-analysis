import cv2
import os
import glob
from tqdm import tqdm
import pickle

class FeatureExtractor:
    def __init__(self, base_path='data/images/paris'):
        self.base_path = base_path

    def extract_features(self):
        if os.path.exists("data/features/sift_features.pkl") and os.path.exists("data/features/hog_features.pkl"):
            print("Características ya extraídas previamente.")
            return

        print("Buscando imágenes...")
        image_files = glob.glob(os.path.join(self.base_path, "**", "*.jpg"), recursive=True)
        print(f"Total de imágenes encontradas: {len(image_files)}")

        print("Iniciando extracción de características (SIFT y HOG)...")

        sift_features = self.extract_sift_features(image_files)
        hog_features = self.extract_hog_features(image_files)

        print("Guardando características en archivos .pkl...")

        os.makedirs('data/features', exist_ok=True)
        with open('data/features/sift_features.pkl', 'wb') as f:
            pickle.dump(sift_features, f)

        with open('data/features/hog_features.pkl', 'wb') as f:
            pickle.dump(hog_features, f)

        print("¡Extracción completada con éxito!")
        print(f"SIFT extraído de: {len(sift_features)} imágenes")
        print(f"HOG extraído de: {len(hog_features)} imágenes")

    def extract_sift_features(self, image_files, nfeatures=500):
        sift_features = {}
        sift = cv2.SIFT_create(nfeatures)

        for path in tqdm(image_files):
            img = cv2.imread(path)
            if img is None:
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, des_sift = sift.detectAndCompute(gray, None)
            if des_sift is not None:
                sift_features[path] = des_sift
        
        return sift_features

    def extract_hog_features(self, image_files, image_size=(128, 128)):
        hog_features = {}
        hog = cv2.HOGDescriptor(
            _winSize=image_size,
            _blockSize=(16, 16),
            _blockStride=(8, 8),
            _cellSize=(8, 8),
            _nbins=9
        )

        for path in tqdm(image_files):
            img = cv2.imread(path)
            if img is None:
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_resized = cv2.resize(gray, image_size)
            des_hog = hog.compute(gray_resized)
            hog_features[path] = des_hog.flatten()

        return hog_features