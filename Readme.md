# Handcrafted Models

## Setup & Execution

1. Place the dataset files:
   - `paris_1.tgz.zip`
   - `paris_2.tgz.zip`  
   inside the folder:
   ```data/raw/```
2. Install dependencies:
```pip install -r requirements.txt```
3. Run the main script:
```python main.py```
---

The script will:
- Extract the dataset
- Compute features (SIFT and HOG)
- Build the Bag of Visual Words (K=100)
- Perform image retrieval on a sample query
