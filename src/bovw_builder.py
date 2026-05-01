import pickle
import os
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt

class BoVWBuilder:
    def build_bovw(self, K=100):
        if os.path.exists("data/features/bovw_histograms.pkl"):
            print("Histogramas BoVW ya generados previamente.")
            return

        print("Cargando descriptores SIFT...")
        with open('data/features/sift_features.pkl', 'rb') as f:
            sift_features = pickle.load(f)

        print("Preparando muestra para K-Means...")
        kmeans, K = self.train_kmeans(sift_features, K=K)

        print("Generando histogramas (Bag of Visual Words)...")
        bovw_histograms = self.build_histograms(sift_features, kmeans, K)

        print("Ploteando histogramas de ejemplo...")
        ejemplos = list(bovw_histograms.keys())[:3]
        for p in ejemplos:
            self.plot_image_and_hist(p, bovw_histograms[p], K)

        with open('data/features/bovw_histograms.pkl', 'wb') as f:
            pickle.dump(bovw_histograms, f)

    def train_kmeans(self, sift_features, K):
        all_des_list = list(sift_features.values())
        descriptors_stack = np.vstack([d for d in all_des_list if d is not None])
        descriptors_sample = descriptors_stack[::10]

        print(f"Entrenando K-Means con K={K} (Esto puede tardar un poco)...")
        kmeans = MiniBatchKMeans(n_clusters=K, n_init=3, random_state=42, batch_size=1000)
        kmeans.fit(descriptors_sample)

        return kmeans, K

    def build_histograms(self, sift_features, kmeans, K):
        bovw_histograms = {}
        for path, des in sift_features.items():
            if des is not None:
                bovw_histograms[path] = self.build_histogram(des, kmeans, K)
        return bovw_histograms

    def build_histogram(self, descriptors, kmeans_model, k_size):
        visual_words = kmeans_model.predict(descriptors)
        hist, _ = np.histogram(visual_words, bins=range(k_size + 1), density=True)
        return hist

    def plot_image_and_hist(self, img_path, histogram, K):
        img = plt.imread(img_path)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))

        ax1.imshow(img)
        ax1.set_title("Imagen Original")
        ax1.axis('off')

        ax2.bar(range(len(histogram)), histogram, color='blue', alpha=0.7)
        ax2.set_title(f"Histograma BoVW (Vocabulario de {K} palabras)")
        ax2.set_xlabel("ID de la Palabra Visual")
        ax2.set_ylabel("Frecuencia (Normalizada)")

        plt.tight_layout()
        plt.show()