import pickle
import numpy as np
import os
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

class VocabularyTuner:
    def vocabulary_tuning(self, k_values=[100, 500, 1000, 2000]):
        with open('data/features/sift_features.pkl', 'rb') as f:
            sift_features = pickle.load(f)

        image_paths = list(sift_features.keys())
        labels = [os.path.basename(os.path.dirname(p)) for p in image_paths]

        results_precision = []
        all_des = np.vstack([d for d in sift_features.values() if d is not None])[::15]

        print("Iniciando experimentación de tamaños de vocabulario...")

        for k in k_values:
            print(f"\n--- Probando K = {k} ---")

            kmeans = MiniBatchKMeans(n_clusters=k, n_init=3, random_state=42)
            kmeans.fit(all_des)

            hists = self.build_histograms_for_k(image_paths, sift_features, kmeans, k)
            avg_p = self.evaluate_precision_at_5(image_paths, labels, hists)

            results_precision.append(avg_p)
            print(f"Precision@5 para K={k}: {avg_p:.4f}")

        self.plot_k_results(k_values, results_precision)

    def build_histograms_for_k(self, image_paths, sift_features, kmeans, k):
        hists = []
        for p in image_paths:
            des = sift_features[p]
            v_words = kmeans.predict(des)
            h, _ = np.histogram(v_words, bins=range(k + 1), density=True)
            hists.append(h)
        return np.array(hists)

    def evaluate_precision_at_5(self, image_paths, labels, hists):
        test_indices = np.random.choice(len(image_paths), 20, replace=False)
        precisions = []

        for idx in test_indices:
            query_hist = hists[idx].reshape(1, -1)
            query_label = labels[idx]

            sims = cosine_similarity(query_hist, hists).flatten()
            best_idx = np.argsort(sims)[::-1][1:6]

            hits = sum([1 for i in best_idx if labels[i] == query_label])
            precisions.append(hits / 5.0)

        return np.mean(precisions)

    def plot_k_results(self, k_values, results_precision):
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, results_precision, marker='o', linestyle='-', color='red', linewidth=2)
        plt.title("Justificación Experimental del Tamaño del Vocabulario (K)")
        plt.xlabel("Tamaño del Vocabulario (Número de Clusters)")
        plt.ylabel("Precision@5 Promedio")
        plt.grid(True, linestyle='--', alpha=0.7)

        for i, val in enumerate(results_precision):
            plt.text(k_values[i], val + 0.01, f"{val:.2f}", ha='center')

        plt.show()