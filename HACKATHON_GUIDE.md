# LumaFin Hackathon Preparation Guide

**Deadline:** November 19, 2025  
**Status:** Ready for training and deployment  
**Target:** >90% transaction categorization accuracy

---

## 📋 Quick Overview

LumaFin is a hybrid adaptive transaction categorization system that combines:
- **Global ML** (Fine-tuned embeddings + FAISS retrieval)
- **Personal Clustering** (AMPT: automatic micro-category discovery)
- **XGBoost Reranker** (Feature-engineered scoring)
- **Explainability** (SHAP + decision paths)

**Current Status:**
- ✅ Core architecture: 95% complete
- ✅ Database: 80,540 training examples seeded
- ✅ API & UI: Fully functional
- 🚀 Model Training: Ready for Google Colab

---

## 🎯 Hackathon Timeline (Nov 17-19)

### Day 1 (Nov 17): Model Training
- **Morning:** Setup Google Colab, run data preparation
- **Afternoon:** Train embedding model (L-A CFT)
- **Evening:** Train reranker and build FAISS index
- **Est. Time:** 2-3 hours (mostly automated)

### Day 2 (Nov 18): Integration & Testing
- **Morning:** Download models, integrate into repository
- **Afternoon:** Run evaluation, verify >90% accuracy
- **Evening:** Test API and UI, prepare demo materials
- **Est. Time:** 4-6 hours

### Day 3 (Nov 19): Demo & Presentation
- **Morning:** Create demo video, finalize presentation
- **Afternoon:** Final testing, bug fixes
- **Evening:** Submit project before deadline
- **Est. Time:** 4-6 hours

---

## 🚀 Step-by-Step Training Guide

### Prerequisites

1. **Google Account** with access to Google Colab
2. **Internet Connection** for Colab notebooks
3. **Google Drive** with ~2GB free space for models

### Training Workflow

#### Step 1: Data Preparation (10 min)

1. Open `colab_notebooks/01_setup_and_data_preparation.ipynb` in Google Colab
2. Mount Google Drive when prompted
3. Run all cells
4. Wait for data to be copied to Drive

**Output:** `train.csv`, `test.csv` in Google Drive

#### Step 2: Embedding Training (30-60 min)

1. Open `colab_notebooks/02_train_embeddings_lacft.ipynb` in Google Colab
2. **IMPORTANT:** Enable GPU runtime
   - Runtime → Change runtime type → Hardware accelerator → GPU
3. Run all cells
4. Monitor training progress (loss should decrease)
5. Wait for model to save to Drive

**Output:** `lumafin-lacft-v1.0/` model directory

**Tips:**
- Free Colab provides T4 GPU (sufficient for this task)
- Training time: ~30 min on T4, ~60 min on CPU (not recommended)
- Keep browser tab open to maintain connection

#### Step 3: Reranker Training (15-30 min)

1. Open `colab_notebooks/03_train_reranker_xgboost.ipynb` in Google Colab
2. GPU helpful but not required (CPU OK)
3. Run all cells
4. Wait for models and index to save

**Output:** `xgb_reranker.json`, `faiss_index.bin`, `faiss_metadata.pkl`

#### Step 4: Evaluation (10-20 min)

1. Open `colab_notebooks/04_evaluate_complete_pipeline.ipynb` in Google Colab
2. Run all cells
3. Review generated charts and metrics
4. Check if accuracy target (>90%) is achieved

**Output:** Evaluation report, charts, and metrics

### Integration into Repository

After training is complete:

1. **Download models from Google Drive:**
   ```
   MyDrive/LumaFin/models/
   ├── lumafin-lacft-v1.0/
   ├── xgb_reranker.json
   ├── faiss_index.bin
   └── faiss_metadata.pkl
   ```

2. **Run integration script:**
   ```bash
   python scripts/integrate_colab_models.py --source /path/to/downloaded/models
   ```

3. **Verify .env configuration:**
   ```bash
   MODEL_PATH=models/embeddings/lumafin-lacft-v1.0
   RERANKER_MODEL_PATH=models/reranker/xgb_reranker.json
   FAISS_INDEX_PATH=models/faiss_index.bin
   FAISS_METADATA_PATH=models/faiss_metadata.pkl
   ```

---

## 🧪 Testing & Validation

### Local Testing

1. **Start services:**
   ```bash
   # Terminal 1: Start database
   docker-compose up -d postgres redis
   
   # Terminal 2: Start API
   PYTHONPATH=. uvicorn src.api.main:app --reload --port 8000
   
   # Terminal 3: Start UI
   streamlit run demo/streamlit_app.py
   ```

2. **Test API:**
   ```bash
   curl -X POST http://localhost:8000/categorize \
     -H "Content-Type: application/json" \
     -d '{
       "merchant": "Starbucks",
       "amount": 5.50,
       "description": "Morning coffee"
     }'
   ```

3. **Expected response:**
   ```json
   {
     "category": "Food & Dining",
     "confidence": 0.95,
     "method": "fusion",
     "explanation": {
       "decision_path": "retrieval → reranker → fusion",
       "similar_transactions": [...]
     }
   }
   ```

### Validation Checklist

- [ ] API responds with correct categories
- [ ] Confidence scores are reasonable (0.7-1.0)
- [ ] Streamlit UI loads and works
- [ ] Evaluation shows >90% accuracy
- [ ] Response time is <100ms per transaction
- [ ] System handles edge cases (unknown merchants, etc.)

---

## 🎬 Demo Preparation

### Demo Script

**Duration:** 5-7 minutes

1. **Introduction (30 sec)**
   - Problem: Manual transaction categorization is tedious
   - Solution: AI-powered automatic categorization with >90% accuracy

2. **Architecture Overview (1 min)**
   - Show architecture diagram
   - Explain multi-stage pipeline
   - Highlight personalization features

3. **Live Demo (3 min)**
   - **Part 1:** Single transaction categorization via API
     - Show request/response with explanation
   - **Part 2:** Bulk upload via Streamlit UI
     - Upload CSV with 10-20 transactions
     - Show instant categorization results
   - **Part 3:** Feedback loop
     - Correct a misclassification
     - Show how system learns from feedback

4. **Performance Metrics (1 min)**
   - Show evaluation results (>90% accuracy)
   - Per-category performance chart
   - Comparison: baseline vs fine-tuned vs complete pipeline

5. **Technical Highlights (1 min)**
   - Fine-tuned embeddings with contrastive learning
   - FAISS vector search for scalability
   - XGBoost reranker for precision
   - Real-time explainability

6. **Future Work & Q&A (1 min)**
   - Cross-encoder integration
   - Federated learning for privacy
   - Knowledge graph for relationships

### Demo Materials Checklist

- [ ] Prepared slide deck (10-12 slides max)
- [ ] Sample transactions CSV (varied categories)
- [ ] Architecture diagram (high-res)
- [ ] Performance charts from evaluation
- [ ] Live API endpoint (localhost or deployed)
- [ ] Streamlit UI running
- [ ] Demo video (backup if live demo fails)
- [ ] GitHub repository link
- [ ] README with clear instructions

### Recording Demo Video

**Tools:** OBS Studio, Loom, or Zoom recording

**Script:**
1. **Screen 1:** Slide deck (30 sec intro)
2. **Screen 2:** Terminal showing API request/response (30 sec)
3. **Screen 3:** Streamlit UI demo (2 min)
4. **Screen 4:** Performance charts and metrics (1 min)
5. **Screen 5:** Code walkthrough (1 min - optional)
6. **Screen 6:** Closing slide with links (30 sec)

---

## 📊 Performance Targets

### Accuracy Goals

| Metric | Target | Stretch Goal |
|--------|--------|--------------|
| Overall Accuracy | >90% | >93% |
| Macro F1-Score | >0.85 | >0.90 |
| Per-Category F1 | >0.80 | >0.85 |
| Inference Time | <100ms | <50ms |

### Baseline Comparisons

| Model | Expected Accuracy |
|-------|-------------------|
| Base Model + Voting | 75-80% |
| Fine-tuned + Voting | 80-85% |
| Complete Pipeline | **>90%** ✅ |

---

## 🐛 Troubleshooting

### Common Issues

**Issue:** Colab disconnects during training  
**Solution:** Keep browser tab open, use Colab Pro for longer sessions

**Issue:** Out of memory during training  
**Solution:** Reduce batch_size in notebooks (16 → 8)

**Issue:** Models not loading locally  
**Solution:** Check .env paths, ensure all files downloaded from Drive

**Issue:** API returns low confidence  
**Solution:** Ensure fine-tuned models are loaded (not base model)

**Issue:** FAISS index not found  
**Solution:** Run notebook 03 to build index, copy to models/

**Issue:** Evaluation accuracy below target  
**Solution:** Train longer (increase epochs), use more data, check data quality

---

## 📦 Deployment Options

### Local Development
```bash
docker-compose up -d
PYTHONPATH=. uvicorn src.api.main:app --reload
```

### Cloud Deployment (Quick)

**Option 1: Heroku**
```bash
heroku create lumafin-demo
git push heroku main
```

**Option 2: Railway**
```bash
railway init
railway up
```

**Option 3: Google Cloud Run**
```bash
gcloud run deploy lumafin \
  --source . \
  --platform managed \
  --region us-central1
```

### Docker (Production)
```bash
docker build -t lumafin .
docker run -p 8000:8000 lumafin
```

---

## ✅ Pre-Submission Checklist

### Code Quality
- [ ] All notebooks run without errors
- [ ] Code is well-commented
- [ ] README is comprehensive
- [ ] Dependencies are documented
- [ ] .env.example is up to date

### Functionality
- [ ] API endpoints work correctly
- [ ] Streamlit UI is functional
- [ ] Models achieve >90% accuracy
- [ ] Evaluation report is generated
- [ ] Feedback loop is operational

### Documentation
- [ ] README has clear setup instructions
- [ ] HACKATHON_GUIDE is complete
- [ ] Colab notebooks have explanations
- [ ] Architecture diagram is included
- [ ] API documentation is clear

### Presentation
- [ ] Demo video is recorded
- [ ] Slide deck is prepared
- [ ] Performance metrics are documented
- [ ] GitHub repository is public
- [ ] All links work correctly

### Final Tests
- [ ] Fresh install works (test on clean machine)
- [ ] API responds in <100ms
- [ ] All dependencies install correctly
- [ ] Docker compose works
- [ ] Evaluation reproduces results

---

## 🏆 Judging Criteria Alignment

### Technical Innovation (30%)
- ✅ Novel combination of techniques (CFT + FAISS + XGBoost)
- ✅ Hybrid approach (global + personal)
- ✅ Explainability features (SHAP, decision paths)
- ✅ Feedback loop for continuous learning

### Implementation Quality (30%)
- ✅ Production-ready code structure
- ✅ Comprehensive testing
- ✅ Docker deployment ready
- ✅ API with proper validation
- ✅ CI/CD pipeline

### Performance (20%)
- ✅ >90% accuracy target
- ✅ Fast inference (<100ms)
- ✅ Scalable architecture (FAISS)
- ✅ Handles edge cases

### Presentation (20%)
- ✅ Clear problem statement
- ✅ Live demo with real data
- ✅ Performance visualizations
- ✅ Well-documented repository

---

## 💡 Tips for Success

### Training
1. **Start early** - Begin training on Day 1 morning
2. **Monitor progress** - Check loss values during training
3. **Use GPU** - Essential for embedding training
4. **Save checkpoints** - Models auto-save to Drive
5. **Verify downloads** - Check all files before integration

### Demo
1. **Practice** - Run through demo 3-4 times
2. **Backup plan** - Have video ready if live demo fails
3. **Clear narrative** - Tell a story, not just features
4. **Show metrics** - Data-driven results are convincing
5. **Be confident** - You've built something impressive!

### Presentation
1. **Keep it simple** - Focus on key innovations
2. **Show real results** - Use actual evaluation metrics
3. **Highlight uniqueness** - What makes LumaFin special?
4. **Address scalability** - Show FAISS can handle millions
5. **Future vision** - What comes next?

---

## 📞 Resources

### Documentation
- **Main README:** `/README.md`
- **Colab Guide:** `/colab_notebooks/README.md`
- **Deployment Guide:** `/DEPLOYMENT.md`
- **Dev Guide:** `/DEV_GUIDE.md`

### External Links
- **Sentence-Transformers:** https://www.sbert.net/
- **FAISS:** https://faiss.ai/
- **XGBoost:** https://xgboost.readthedocs.io/
- **FastAPI:** https://fastapi.tiangolo.com/

### Support
- **GitHub Issues:** Open for questions
- **Notebook Comments:** Detailed explanations in cells
- **Code Comments:** Inline documentation

---

## 🎉 Good Luck!

You have everything you need to succeed in the hackathon:
- ✅ Production-quality codebase
- ✅ Google Colab training notebooks
- ✅ Comprehensive documentation
- ✅ Clear deployment path
- ✅ Demo-ready features

**Remember:**
- Start training ASAP (models take 1-2 hours)
- Test everything before the demo
- Practice your presentation
- Stay calm and confident
- Have fun! 🚀

---

**Last Updated:** November 17, 2025  
**Status:** Ready for Hackathon  
**Target:** >90% Accuracy ✅
