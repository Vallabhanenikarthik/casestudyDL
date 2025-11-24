"""
Fake News Detection — BERT + TF-IDF Hybrid
Single-file project you can open in VS Code.

Files included (all inside this single script for convenience):
- requirements.txt (content shown below)
- README (brief run instructions shown below)
- train_hybrid.py (this file): performs data loading, TF-IDF+LR training, BERT fine-tuning,
  and trains a small meta-classifier that ensembles the two models' predicted probabilities.

Notes:
- Uses the dataset files already uploaded to your environment:
  /mnt/data/True.csv  (real news)
  /mnt/data/Fake.csv  (fake news)
- Will use a lightweight BERT (distilbert) so it's faster for experimentation.
- If you don't have a GPU, training BERT will be slow. You can skip BERT and only use TF-IDF flow
  by passing --no-bert.

--- requirements.txt ---
# pip install -r requirements.txt
pandas
numpy
scikit-learn
transformers>=4.0.0
datasets
torch
tqdm
joblib

--- How to run (quick) ---
1. Create and activate a virtual environment in VS Code (terminal):
   python -m venv .venv
   source .venv/bin/activate      # Linux / macOS
   .venv\Scripts\activate       # Windows PowerShell

2. Install dependencies:
   pip install -r requirements.txt

3. Run the single script for full pipeline (may take a while if BERT is enabled):
   python fake_news_bert_tfidf_project.py --out_dir ./model_output --do_all

4. To run only TF-IDF baseline (fast):
   python fake_news_bert_tfidf_project.py --no_bert --out_dir ./model_output

--- train_hybrid.py (this file) ---
"""
import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

# BERT imports (HuggingFace)
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict


def load_csv_datasets(true_path, fake_path):
    # Both CSVs assumed to have a 'title' and 'text' column or at least 'text'.
    df_true = pd.read_csv(true_path)
    df_fake = pd.read_csv(fake_path)

    # unify column names
    # prefer 'text' if exists, else combine title + text
    def _get_text(row):
        if 'text' in row and pd.notna(row['text']):
            return str(row['text'])
        t = ''
        if 'title' in row and pd.notna(row['title']):
            t += str(row['title'])
        if 'description' in row and pd.notna(row['description']):
            t += ' ' + str(row['description'])
        return t.strip()

    df_true['text'] = df_true.apply(_get_text, axis=1)
    df_fake['text'] = df_fake.apply(_get_text, axis=1)

    df_true = df_true[['text']].copy()
    df_fake = df_fake[['text']].copy()

    df_true['label'] = 0  # real
    df_fake['label'] = 1  # fake

    df = pd.concat([df_true, df_fake], ignore_index=True)
    df = df.dropna(subset=['text']).reset_index(drop=True)
    return df


def train_tfidf_model(X_train, y_train, X_val, y_val, out_dir):
    print('\n[TF-IDF] training...')
    vect = TfidfVectorizer(max_features=20000, ngram_range=(1,2), stop_words='english')
    X_train_tfidf = vect.fit_transform(X_train)
    X_val_tfidf = vect.transform(X_val)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_tfidf, y_train)

    preds = clf.predict(X_val_tfidf)
    probs = clf.predict_proba(X_val_tfidf)[:,1]

    print('[TF-IDF] val acc:', accuracy_score(y_val, preds))
    print('[TF-IDF] classification report:\n', classification_report(y_val, preds))

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    dump({'vectorizer': vect, 'clf': clf}, os.path.join(out_dir, 'tfidf_lr.joblib'))
    print('[TF-IDF] saved to', out_dir)
    return probs, preds


# --- BERT training / predict helpers ---

def prepare_hf_dataset(texts, labels, tokenizer, max_length=256):
    enc = tokenizer(texts, truncation=True, padding='max_length', max_length=max_length)
    ds = Dataset.from_dict({
        'input_ids': enc['input_ids'],
        'attention_mask': enc['attention_mask'],
        'labels': labels
    })
    return ds


def hf_train_and_predict(train_texts, train_labels, val_texts, val_labels, out_dir, model_name='distilbert-base-uncased', epochs=2, batch_size=8):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[BERT] device = {device}, model = {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    train_ds = prepare_hf_dataset(train_texts, train_labels, tokenizer)
    val_ds = prepare_hf_dataset(val_texts, val_labels, tokenizer)

    dataset = DatasetDict({'train': train_ds, 'validation': val_ds})

    training_args = TrainingArguments(
    output_dir=os.path.join(out_dir, 'hf_checkpoint'),
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    logging_steps=50,
    save_total_limit=2
)


    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:,1]
        return {
            'accuracy': (preds == labels).mean(),
            'roc_auc': roc_auc_score(labels, probs)
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # Predict on validation set
    preds_out = trainer.predict(dataset['validation'])
    logits = preds_out.predictions
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:,1]
    preds = np.argmax(logits, axis=1)

    # save model and tokenizer
    model.save_pretrained(os.path.join(out_dir, 'hf_model'))
    tokenizer.save_pretrained(os.path.join(out_dir, 'hf_tokenizer'))
    print('[BERT] saved to', os.path.join(out_dir, 'hf_model'))
    print('[BERT] val acc:', accuracy_score(val_labels, preds))
    print('[BERT] classification report:\n', classification_report(val_labels, preds))
    return probs, preds


def train_meta_classifier(probs_tfidf, probs_bert, y_val, out_dir):
    print('\n[Meta] training meta-classifier on probs...')
    X_meta = np.vstack([probs_tfidf, probs_bert]).T
    meta = LogisticRegression()
    meta.fit(X_meta, y_val)
    preds = meta.predict(X_meta)
    probs = meta.predict_proba(X_meta)[:,1]
    print('[Meta] val acc:', accuracy_score(y_val, preds))
    print('[Meta] classification report:\n', classification_report(y_val, preds))
    dump(meta, os.path.join(out_dir, 'meta_lr.joblib'))
    return meta


def main(args):
    # load data
    df = load_csv_datasets(args.true_csv, args.fake_csv)
    print('Total samples loaded:', len(df))

    # simple preprocessing
    df['text'] = df['text'].astype(str)

    X = df['text'].values
    y = df['label'].values

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    out_dir = args.out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # TF-IDF baseline
    probs_tfidf_val, preds_tfidf_val = train_tfidf_model(X_train, y_train, X_val, y_val, out_dir)

    probs_bert_val = None
    preds_bert_val = None

    if not args.no_bert:
        probs_bert_val, preds_bert_val = hf_train_and_predict(X_train.tolist(), y_train.tolist(), X_val.tolist(), y_val.tolist(), out_dir, model_name=args.bert_model, epochs=args.epochs, batch_size=args.batch_size)
    else:
        # if BERT disabled, use TF-IDF probs as second input (weak)
        probs_bert_val = probs_tfidf_val

    # train meta-classifier on validation set outputs
    meta = train_meta_classifier(probs_tfidf_val, probs_bert_val, y_val, out_dir)

    # Evaluate on test set: produce probs from saved TF-IDF + BERT models
    print('\n[Evaluation] running on test set...')
    # load TF-IDF
    tfobj = load(os.path.join(out_dir, 'tfidf_lr.joblib'))
    vect = tfobj['vectorizer']; clf = tfobj['clf']
    X_test_tfidf = vect.transform(X_test)
    probs_tfidf_test = clf.predict_proba(X_test_tfidf)[:,1]

    if not args.no_bert:
        # load HF tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(out_dir, 'hf_tokenizer'))
        model = AutoModelForSequenceClassification.from_pretrained(os.path.join(out_dir, 'hf_model'))
        model.to('cuda' if torch.cuda.is_available() else 'cpu')

        enc = tokenizer(list(X_test), truncation=True, padding=True, max_length=256, return_tensors='pt')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
            logits = out.logits.cpu().numpy()
            probs_bert_test = torch.softmax(torch.tensor(logits), dim=1).numpy()[:,1]
    else:
        probs_bert_test = probs_tfidf_test

    X_meta_test = np.vstack([probs_tfidf_test, probs_bert_test]).T
    preds_meta_test = meta.predict(X_meta_test)
    probs_meta_test = meta.predict_proba(X_meta_test)[:,1]

    print('[Final Ensemble] test acc:', accuracy_score(y_test, preds_meta_test))
    print('[Final Ensemble] classification report:\n', classification_report(y_test, preds_meta_test))

    # save a small CSV with predictions
    out_df = pd.DataFrame({'text': X_test, 'label': y_test, 'pred': preds_meta_test, 'prob': probs_meta_test})
    out_df.to_csv(os.path.join(out_dir, 'test_predictions.csv'), index=False)
    print('[Done] outputs saved to', out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--true_csv', type=str, default='/mnt/data/True.csv', help='Path to True.csv (real news)')
    parser.add_argument('--fake_csv', type=str, default='/mnt/data/Fake.csv', help='Path to Fake.csv (fake news)')
    parser.add_argument('--out_dir', type=str, default='./model_output', help='Where to save models and outputs')
    parser.add_argument('--no_bert', action='store_true', help='Skip BERT training/prediction (only TF-IDF)')
    parser.add_argument('--bert_model', type=str, default='distilbert-base-uncased', help='HuggingFace model name')
    parser.add_argument('--epochs', type=int, default=2, help='Epochs for BERT fine-tuning')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for BERT')
    parser.add_argument('--do_all', action='store_true', help='Alias to do full run')
    args = parser.parse_args()

    main(args)
