import cv2
import numpy as np
import os
import pickle
import glob
from skimage import feature
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from tqdm import tqdm

class ImageRetrieval:
    def __init__(self):
        self.BASE_PATH = 'data/images/paris'
        self.HOG_PKL = 'data/features/hog_features.pkl'
        self.LBP_PKL = 'data/features/lbp_features.pkl'
        self.BOVW_PKL = 'data/features/bovw_histograms.pkl'

    def extract_lbp_single(self, image_path):
        img = cv2.imread(image_path)
        if img is None: return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (256, 256))
        
        lbp = feature.local_binary_pattern(gray, 24, 3, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        return hist

    def extract_lbp_features(self):
        image_files = glob.glob(os.path.join(self.BASE_PATH, "**", "*.jpg"), recursive=True)

        if not os.path.exists(self.LBP_PKL):
            print("Extrayendo LBP de todas las imágenes...")
            lbp_features = {}
            for path in tqdm(image_files):
                feat = self.extract_lbp_single(path)
                if feat is not None: lbp_features[path] = feat
            
            with open(self.LBP_PKL, 'wb') as f:
                pickle.dump(lbp_features, f)
            print(f"LBP guardado en {self.LBP_PKL}")
        else:
            print("Cargando LBP desde archivo...")
            with open(self.LBP_PKL, 'rb') as f: lbp_features = pickle.load(f)

        with open(self.HOG_PKL, 'rb') as f: hog_features = pickle.load(f)
        with open(self.BOVW_PKL, 'rb') as f: bovw_histograms = pickle.load(f)

        return lbp_features, hog_features, bovw_histograms

    def get_super_vector(self, path, bovw_histograms, hog_features, lbp_features):
        v_sift = bovw_histograms[path]
        v_hog = hog_features[path]
        v_lbp = lbp_features[path]
        
        n_sift = v_sift / (np.linalg.norm(v_sift) + 1e-7)
        n_hog = v_hog / (np.linalg.norm(v_hog) + 1e-7)
        n_lbp = v_lbp / (np.linalg.norm(v_lbp) + 1e-7)
        
        return np.hstack([n_sift, n_hog, n_lbp])

    def test_improved_search(self, query_path):
        lbp_features, hog_features, bovw_histograms = self.extract_lbp_features()

        print("Construyendo base de datos con Super Vectores...")
        valid_paths = list(bovw_histograms.keys())
        super_db = np.array([self.get_super_vector(p, bovw_histograms, hog_features, lbp_features) for p in valid_paths])

        q_vec = self.get_super_vector(query_path, bovw_histograms, hog_features, lbp_features).reshape(1, -1)
        sims = cosine_similarity(q_vec, super_db).flatten()
        indices = np.argsort(sims)[::-1][:5]

        plt.figure(figsize=(18, 5))
        plt.suptitle(f"Búsqueda con SUPER VECTOR (SIFT + HOG + LBP)\nQuery: {os.path.basename(query_path)}", fontsize=15)
        
        for i, idx in enumerate(indices):
            img_res = cv2.cvtColor(cv2.imread(valid_paths[idx]), cv2.COLOR_BGR2RGB)
            plt.subplot(1, 5, i+1)
            plt.imshow(img_res)
            plt.title(f"Top {i+1}\nSim: {sims[idx]:.3f}")
            plt.axis('off')
        plt.show()