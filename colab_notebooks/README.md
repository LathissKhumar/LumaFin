# LumaFin Google Colab Training Notebooks

This directory contains Google Colab notebooks for training LumaFin models without local GPU access.

## 📚 Notebooks Overview

### 1. `01_setup_and_data_preparation.ipynb`
**Purpose:** Prepares training data and saves to Google Drive  
**Runtime:** CPU (no GPU needed)  
**Time:** ~10 minutes  
**Output:**
- `train.csv` - Training dataset
- `test.csv` - Test dataset
- `category_distribution.png` - Data visualization

### 2. `02_train_embeddings_lacft.ipynb`
**Purpose:** Fine-tunes sentence-transformers model with Label-Aware Contrastive Fine-Tuning  
**Runtime:** GPU Required (T4 or better)  
**Time:** ~30-60 minutes  
**Output:**
- `lumafin-lacft-v1.0/` - Fine-tuned embedding model

### 3. `03_train_reranker_xgboost.ipynb`
**Purpose:** Trains XGBoost reranker and builds FAISS index  
**Runtime:** GPU helpful but CPU OK  
**Time:** ~15-30 minutes  
**Output:**
- `xgb_reranker.pkl` - Calibrated XGBoost classifier
- `xgb_reranker.json` - XGBoost model (JSON format)
- `faiss_index.bin` - FAISS vector index
- `faiss_metadata.pkl` - Index metadata

### 4. `04_evaluate_complete_pipeline.ipynb`
**Purpose:** Evaluates complete pipeline and generates performance reports  
**Runtime:** GPU helpful but CPU OK  
**Time:** ~10-20 minutes  
**Output:**
- `evaluation_report.md` - Complete evaluation summary
- `results_comparison.png` - Model comparison chart
- `confusion_matrix.png` - Confusion matrix heatmap
- `per_category_f1.png` - Per-category performance
- `per_category_metrics.csv` - Detailed metrics

## 🚀 Quick Start Guide

### Step 1: Upload Notebooks to Google Colab

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload each notebook from this directory
3. Or use direct GitHub import: `File → Open notebook → GitHub` and enter the repository URL

### Step 2: Run Notebooks in Order

**⚠️ IMPORTANT:** Run notebooks sequentially (01 → 02 → 03 → 04)

#### Notebook 01: Data Preparation
```
Runtime: Python 3, CPU
1. Run all cells
2. Wait for data to be copied to Google Drive
```

#### Notebook 02: Embedding Training
```
Runtime: Python 3, GPU (T4 or better)
⚠️ Enable GPU: Runtime → Change runtime type → GPU
1. Run all cells
2. Training takes 30-60 minutes depending on data size
3. Model will be saved to Google Drive automatically
```

#### Notebook 03: Reranker Training
```
Runtime: Python 3, GPU helpful but not required
1. Run all cells
2. Training takes 15-30 minutes
3. Models and index saved to Google Drive
```

#### Notebook 04: Evaluation
```
Runtime: Python 3, GPU helpful but not required
1. Run all cells
2. Review generated charts and metrics
3. Check evaluation_report.md for summary
```

## 📥 Downloading Trained Models

After training is complete, download models from Google Drive:

### Google Drive Structure
```
MyDrive/
└── LumaFin/
    ├── data/
    │   ├── train.csv
    │   ├── test.csv
    │   └── category_distribution.png
    ├── models/
    │   ├── lumafin-lacft-v1.0/  (embedding model)
    │   ├── xgb_reranker.pkl
    │   ├── xgb_reranker.json
    │   ├── faiss_index.bin
    │   └── faiss_metadata.pkl
    └── results/
        ├── evaluation_report.md
        ├── results_comparison.png
        ├── confusion_matrix.png
        ├── per_category_f1.png
        └── per_category_metrics.csv
```

### Download to Local Repository

1. Download the `models/` folder from Google Drive
2. Place in your local repository:
   ```bash
   LumaFin/
   └── models/
       ├── embeddings/
       │   └── lumafin-lacft-v1.0/
       ├── reranker/
       │   └── xgb_reranker.json
       ├── faiss_index.bin
       └── faiss_metadata.pkl
   ```

3. Update `.env` file:
   ```bash
   MODEL_PATH=models/embeddings/lumafin-lacft-v1.0
   RERANKER_MODEL_PATH=models/reranker/xgb_reranker.json
   FAISS_INDEX_PATH=models/faiss_index.bin
   FAISS_METADATA_PATH=models/faiss_metadata.pkl
   ```

## 🎯 Expected Results

### Target Metrics
- **Baseline (Base + Voting):** 75-80% accuracy
- **Fine-tuned (L-A CFT + Voting):** 80-85% accuracy
- **Complete (Fine-tuned + Reranker):** **>90% accuracy** ✅

### Training Times (on Colab T4 GPU)
- Embedding fine-tuning: 30-60 minutes
- Reranker training: 15-30 minutes
- Evaluation: 10-20 minutes
- **Total: ~1-2 hours**

## 💡 Tips and Tricks

### GPU Runtime
- **Free Tier:** T4 GPU with usage limits
- **Colab Pro:** Better GPUs (V100, A100) and longer runtime
- **Connection:** Keep browser tab open to maintain connection

### Saving Progress
- All models are automatically saved to Google Drive
- You can stop and resume training by re-running cells
- FAISS index is saved incrementally

### Memory Management
If you encounter memory errors:
1. Reduce `batch_size` in training notebooks
2. Use fewer training examples (subset the data)
3. Restart runtime: `Runtime → Restart runtime`

### Troubleshooting

**Issue:** "GPU not available"  
**Solution:** Runtime → Change runtime type → Hardware accelerator → GPU

**Issue:** "Drive quota exceeded"  
**Solution:** Free up space in Google Drive or upgrade storage

**Issue:** "Session disconnected"  
**Solution:** Keep browser tab open. Use Colab Pro for longer sessions.

**Issue:** "Model not found"  
**Solution:** Ensure you ran previous notebooks in order

## 📊 Performance Monitoring

### During Training
- Watch loss values (should decrease)
- Monitor GPU utilization (should be high)
- Check for OOM errors (reduce batch size if needed)

### After Training
- Review evaluation metrics in notebook 04
- Compare baseline vs fine-tuned vs complete pipeline
- Check per-category F1 scores for balanced performance

## 🔄 Retraining

To retrain models with new data:
1. Update `data/merged_training.csv` in the repository
2. Re-run notebook 01 to prepare new splits
3. Re-run notebooks 02, 03, 04 in sequence
4. Download updated models

## 🎓 Learning Resources

### Contrastive Learning
- [Sentence-Transformers Documentation](https://www.sbert.net/)
- [Contrastive Learning Paper](https://arxiv.org/abs/2004.11362)

### XGBoost
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Feature Engineering Guide](https://www.kaggle.com/learn/feature-engineering)

### FAISS
- [FAISS Documentation](https://faiss.ai/)
- [Vector Search Tutorial](https://www.pinecone.io/learn/faiss/)

## 📝 Citation

If you use these notebooks in your work, please cite:

```bibtex
@software{lumafin2025,
  title={LumaFin: Hybrid Adaptive System for Transaction Categorization},
  author={Lathiss Khumar},
  year={2025},
  url={https://github.com/LathissKhumar/LumaFin}
}
```

## 🤝 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review notebook cell outputs for error messages
3. Open an issue on GitHub with error details
4. Include notebook name and cell number in reports

## ✅ Checklist for Hackathon

- [ ] Run notebook 01 (data preparation)
- [ ] Run notebook 02 (embedding training)
- [ ] Run notebook 03 (reranker training)
- [ ] Run notebook 04 (evaluation)
- [ ] Download all trained models
- [ ] Integrate models into local repository
- [ ] Test API with trained models
- [ ] Verify >90% accuracy target
- [ ] Prepare demo presentation
- [ ] Create demo video

---

**Good luck with your hackathon! 🚀**
