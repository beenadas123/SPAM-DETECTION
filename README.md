[README.md](https://github.com/user-attachments/files/33470188/README.md)
# SMS Spam Classification with SVM (Improved)

A notebook that classifies SMS messages as **spam** or **ham** (not spam) using a TF-IDF + Linear SVM pipeline, with model comparison, hyperparameter tuning, threshold tuning and error analysis.

- **Notebook:** `SMS_Spam_SVM_Improved.ipynb`
- **Dataset:** [SMS Spam Collection (Kaggle)](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset), 5,572 messages, about 13% spam

## Quick start (Google Colab)

1. Open https://colab.research.google.com and choose **File → Upload notebook**.
2. Select `SMS_Spam_SVM_Improved.ipynb`.
3. Choose **Runtime → Run all**.
4. When asked, upload `spam.csv` (or `SMS.csv`) from the Kaggle dataset.
5. Download `sms_spam_svm.joblib` before closing the session, because Colab storage is temporary.

## Run locally

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk wordcloud joblib jupyter
jupyter notebook SMS_Spam_SVM_Improved.ipynb
```

Put `spam.csv` or `SMS.csv` in the same folder as the notebook. The first run downloads the NLTK `stopwords` and `wordnet` data.

## Pipeline

| Step | What happens |
|------|--------------|
| 1. Load | Auto-detects `spam.csv` / `SMS.csv`, keeps the `v1` (label) and `v2` (message) columns |
| 2. Quality checks | Missing values, duplicates, class balance; duplicates removed **before** the split |
| 3. EDA | Class counts, message length, word clouds for spam and ham |
| 4. Cleaning | Lowercase, stopword removal, lemmatization; URLs, e-mails, phone numbers, numbers and money symbols are replaced with tokens instead of deleted |
| 5. Split | Stratified 80/20 train/test; the test set is untouched until the end |
| 6. Model comparison | 5-fold CV on the training set for LinearSVC, balanced LinearSVC, Logistic Regression, SGD, Multinomial NB and Complement NB |
| 7. Tuning | `GridSearchCV` on the SVM pipeline (n-grams, `min_df`, `max_features`, `C`, `class_weight`), optimised for spam-class F1 |
| 8. Threshold | Decision threshold chosen from out-of-fold training scores |
| 9. Evaluation | Accuracy, precision, recall, F1, MCC, ROC-AUC, PR-AUC, confusion matrices, ROC and PR curves on the test set |
| 10. Stability | Repeated stratified 5-fold CV on the full dataset (mean ± std) |
| 11. Error analysis | Lists false positives and missed spam |
| 12. Interpretation | Top 15 spam and ham indicator words |
| 13. Deployment | Saves the model and threshold to `sms_spam_svm.joblib`; `predict_sms()` for new messages |

## Baseline from the original notebook

| Metric | Value |
|--------|-------|
| Accuracy | 0.9845 |
| Precision | 0.9832 |
| Recall | 0.8931 |
| F1 | 0.9360 |

The main weakness was recall: about 1 in 9 spam messages was missed. The improved notebook targets this through class weighting, tuning for F1 and threshold selection. Run it on the real dataset to see the new numbers.

## Using the saved model

```python
import joblib

saved = joblib.load("sms_spam_svm.joblib")
model, threshold = saved["model"], saved["threshold"]

# The text must go through the same clean_text() function defined in the notebook
score = model.decision_function([clean_text("You won a free prize! Call now")])[0]
print("Spam" if score > threshold else "Ham")
```

`clean_text` lives in the notebook, so copy it into any script that loads the model.

## Notes and limitations

- The dataset is small and English-only; performance on other languages or newer spam styles may be lower.
- Tuning the threshold usually trades a little precision for better recall. Check both on the test set before choosing.
- The notebook is saved without outputs; running it generates them.
