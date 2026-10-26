# Comment Toxicity Detection (Multi-Label NLP)

Multi-label classifier that detects 6 types of toxicity in online comments:
toxic, severe_toxic, obscene, threat, insult, identity_hate.

## Dataset
-Toxic Comment Classification dataset: [https://drive.google.com/drive/folders/1sCX0F1bL2gHe15uiiHpY7-lGBQqOA94z?usp=sharing]
- 159,571 comments, 6 binary labels
- Highly imbalanced: most comments have no toxic label
- 80/20 train-test split (127,656 train / 31,915 test)
- Dataset is not included in this repo; download `train.csv` from Kaggle

## Approach
1. Text cleaning: lowercasing, HTML tag / URL / digit / special-character removal
2. TF-IDF vectorization (50,000 features, unigrams + bigrams, English stop words removed)
3. One-vs-Rest Logistic Regression, compared with and without `class_weight='balanced'`

## Results (test set)
| Model | Micro F1 | Macro F1 | Macro Precision | Macro Recall |
|---|---|---|---|---|
| Without class balancing | 0.66 | 0.47 | 0.77 | 0.36 |
| With `class_weight='balanced'` | 0.68 | 0.55 | 0.44 | 0.81 |

## Key Observations
- Class balancing raised macro recall from 0.36 to 0.81, at the cost of precision (0.77 to 0.44)
- Accuracy dropped from 91.76% to 87.85% after balancing, which shows accuracy is a misleading metric on this imbalanced dataset, so F1 and recall are reported
- Rare labels (threat: 74 test samples) are the hardest to detect
- Example of limitation: "I hate you" was also flagged as identity_hate (false positive), reflecting the low precision

## Possible Improvements
- Per-label threshold tuning and ROC-AUC reporting
- Character n-grams for obfuscated words
- Transformer models (e.g. DistilBERT)

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, NLTK, spaCy, Matplotlib, Seaborn

## Run
Open the notebook in Colab: [Colab link]
