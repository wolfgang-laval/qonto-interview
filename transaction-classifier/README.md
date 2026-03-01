qonto-categorization/
├── README.md
├── requirements.txt
├── data/
│   └── transactions.csv
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_lgbm.ipynb
│   └── 04_embeddings_model.ipynb
├── src/
│   ├── features.py       # feature engineering (or preprocessing)
│   ├── train.py          # script d'entraînement
│   ├── evaluate.py       # métriques + visualisations
├── models/
│   └── (artefacts sauvegardés)
└── Makefile              # commandes : make train, make evaluate