from src.dataset_extractor import DatasetExtractor
from src.feature_extractor import FeatureExtractor
from src.bovw_builder import BoVWBuilder
from src.vocabulary_tuner import VocabularyTuner
from src.image_retrieval import ImageRetrieval
import os

if __name__ == "__main__":
    DatasetExtractor().extract_dataset()
    FeatureExtractor().extract_features()

    # VocabularyTuner().vocabulary_tuning([50, 100, 200, 400])
    
    BoVWBuilder().build_bovw(K=100)

    query_image = 'data/images/paris/defense/paris_defense_000032.jpg'
    if not os.path.exists(query_image):
        print(f"No se encontró la imagen query en: {query_image}")
    else:
        ImageRetrieval().test_improved_search(query_image)