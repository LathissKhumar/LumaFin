# Google Colab Training - Quick Start

**⚡ Fast track to train LumaFin models without local GPU**

---

## 🚀 3-Step Quick Start

### Step 1: Open Notebooks (2 min)

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click: `File → Upload notebook`
3. Upload these 4 notebooks from `colab_notebooks/`:
   - `01_setup_and_data_preparation.ipynb`
   - `02_train_embeddings_lacft.ipynb`
   - `03_train_reranker_xgboost.ipynb`
   - `04_evaluate_complete_pipeline.ipynb`

### Step 2: Run Training (1-2 hours)

Run each notebook in order:

**Notebook 01** _(10 min, CPU)_
```
Mount Google Drive → Run all cells → Data saved to Drive ✓
```

**Notebook 02** _(30-60 min, GPU REQUIRED)_
```
⚠️ Enable GPU: Runtime → Change runtime type → GPU
Mount Drive → Run all cells → Model saved to Drive ✓
```

**Notebook 03** _(15-30 min, GPU optional)_
```
Mount Drive → Run all cells → Reranker saved to Drive ✓
```

**Notebook 04** _(10-20 min, GPU optional)_
```
Mount Drive → Run all cells → Check accuracy > 90% ✓
```

### Step 3: Download & Integrate (10 min)

1. **Download from Google Drive:**
   ```
   MyDrive/LumaFin/models/ → Download folder
   ```

2. **Run integration script:**
   ```bash
   python scripts/integrate_colab_models.py --source ~/Downloads/models
   ```

3. **Test the system:**
   ```bash
   PYTHONPATH=. uvicorn src.api.main:app --reload
   ```

---

## ⏱️ Timeline

| Task | Time | Runtime |
|------|------|---------|
| Setup & Data | 10 min | CPU |
| Embedding Training | 30-60 min | GPU (T4) |
| Reranker Training | 15-30 min | CPU/GPU |
| Evaluation | 10-20 min | CPU/GPU |
| **Total** | **1-2 hours** | **Colab Free Tier OK** |

---

## 📦 What You Get

After completion:

```
models/
├── lumafin-lacft-v1.0/          # Fine-tuned embeddings
│   ├── config.json
│   ├── pytorch_model.bin
│   └── ...
├── xgb_reranker.json            # Trained reranker
├── faiss_index.bin              # Vector index (80k vectors)
└── faiss_metadata.pkl           # Category metadata

Results:
├── evaluation_report.md         # Performance summary
├── results_comparison.png       # Accuracy chart
├── confusion_matrix.png         # Per-category confusion
└── per_category_metrics.csv     # Detailed metrics
```

---

## 🎯 Expected Results

| Metric | Target | Your Result |
|--------|--------|-------------|
| Overall Accuracy | >90% | ___ % |
| Macro F1-Score | >0.85 | ___ |
| Inference Time | <100ms | ___ ms |

**Baseline Comparison:**
- Base Model: ~75-80%
- Fine-tuned: ~80-85%
- Complete Pipeline: **>90%** ✅

---

## ⚠️ Important Notes

### GPU Runtime
- **Notebook 02 REQUIRES GPU** (embedding training)
- Enable: `Runtime → Change runtime type → Hardware accelerator → GPU`
- Free tier provides T4 GPU (sufficient)
- Training without GPU: ~5-10x slower (not recommended)

### Keep Connection Alive
- Keep browser tab open during training
- Colab disconnects after ~12 hours on free tier
- Use Colab Pro for longer sessions

### Storage
- Models saved to Google Drive automatically
- Requires ~2GB free space in Drive
- Download models before Drive quota expires

---

## 🐛 Quick Troubleshooting

**"GPU not available"**
```
Runtime → Change runtime type → GPU → Save
```

**"Session disconnected"**
```
Keep browser tab open
Use Colab Pro for longer sessions
```

**"Out of memory"**
```
Edit cell: batch_size = 16 → batch_size = 8
Runtime → Restart runtime → Run again
```

**"Models not found"**
```
Check Google Drive: MyDrive/LumaFin/models/
Re-run previous notebook if missing
```

---

## 📋 Checklist

Before starting:
- [ ] Google account ready
- [ ] 2GB+ free in Google Drive
- [ ] Stable internet connection
- [ ] ~2 hours available for training

During training:
- [ ] Notebook 01 completed ✓
- [ ] Notebook 02 completed (GPU enabled) ✓
- [ ] Notebook 03 completed ✓
- [ ] Notebook 04 completed ✓
- [ ] Accuracy > 90% achieved ✓

After training:
- [ ] Models downloaded from Drive
- [ ] Integration script run successfully
- [ ] API tested and working
- [ ] Evaluation report reviewed

---

## 🆘 Need Help?

**Full documentation:** `colab_notebooks/README.md`  
**Hackathon guide:** `HACKATHON_GUIDE.md`  
**Main README:** `README.md`

**Quick test:**
```bash
# After integration
curl -X POST http://localhost:8000/categorize \
  -H "Content-Type: application/json" \
  -d '{"merchant": "Starbucks", "amount": 5.50}'
```

Expected response:
```json
{
  "category": "Food & Dining",
  "confidence": 0.95,
  ...
}
```

---

## ✅ Success Criteria

You're ready when:
1. ✅ All 4 notebooks run without errors
2. ✅ Models downloaded and integrated
3. ✅ API responds with >90% confidence
4. ✅ Evaluation shows >90% accuracy
5. ✅ System ready for demo

---

**Good luck! 🚀**

**Questions?** Check `colab_notebooks/README.md` for detailed guide.
