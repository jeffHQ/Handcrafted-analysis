import zipfile
import tarfile
import os
import shutil

class DatasetExtractor:
    def __init__(self, path='data/images/paris'):
        self.path = path

    def extract_dataset(self):
        if not os.path.exists(self.path):
            print("Dataset no encontrado. Extrayendo...")
            self.extract_nested_datasets()
        else:
            print("Dataset ya extraído.")

    def extract_nested_datasets(self, target_folder='data/images'):
        zip_files = ['data/raw/paris_1.tgz.zip', 'data/raw/paris_2.tgz.zip']

        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        
        for zip_path in zip_files:
            print(f"--- Procesando {zip_path} ---")

            if not os.path.isfile(zip_path):
                print(f"Archivo {zip_path} no encontrado. Saltando...")
                continue
            
            # 1. Extraer el .tgz del .zip
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall('data/temp_tgz')
                
                for file in os.listdir('data/temp_tgz'):
                    if file.endswith('.tgz'):
                        tgz_path = os.path.join('data/temp_tgz', file)
                        print(f"Extraído {file}, ahora descomprimiendo imágenes...")
                        
                        # 2. Extraer las imágenes del .tgz
                        with tarfile.open(tgz_path, "r:gz") as tar:
                            tar.extractall(path=target_folder)
            
            shutil.rmtree('data/temp_tgz')
            print(f"¡Listo! Imágenes de {zip_path} guardadas en '{target_folder}'.")