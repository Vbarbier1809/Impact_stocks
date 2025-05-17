from inspect import signature
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, concatenate_datasets, DatasetDict, ClassLabel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_training_args(output_dir, batch_size=16, num_epochs=3):
    """
    Renvoie des TrainingArguments 100 % compatibles
    avec la version de transformers actuellement chargée.
    """
    params = signature(TrainingArguments.__init__).parameters
    base_kwargs = dict(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
    )
    if "evaluation_strategy" in params:
        base_kwargs.update(
            evaluation_strategy="epoch",
            save_strategy      ="epoch"  if "save_strategy"      in params else None,
            logging_strategy   ="epoch"  if "logging_strategy"   in params else None,
            load_best_model_at_end=True  if "load_best_model_at_end" in params else None,
            metric_for_best_model="accuracy" if "metric_for_best_model" in params else None,
            report_to          ="none"   if "report_to"          in params else None,
        )
    elif "eval_strategy" in params:
        base_kwargs.update(
            eval_strategy     ="epoch",
            save_strategy     ="epoch" if "save_strategy"   in params else None,
            logging_strategy  ="epoch" if "logging_strategy" in params else None,
            load_best_model_at_end=True if "load_best_model_at_end" in params else None,
            metric_for_best_model="accuracy" if "metric_for_best_model" in params else None,
            report_to         ="none" if "report_to"         in params else None,
        )
    else:
        base_kwargs.update(
            do_train=True,
            do_eval=True,
            evaluate_during_training=True,
            logging_steps=100,
            save_steps=0,
        )

    final_kwargs = {k: v for k, v in base_kwargs.items() if k in params and v is not None}

    return TrainingArguments(**final_kwargs)

def load_and_prepare_datasets(test_size: float = 0.2, seed: int = 42) -> DatasetDict:
    print("🔹 Chargement des datasets…")
    ds1 = load_dataset(
        "zeroshot/twitter-financial-news-sentiment",
        split="train"
    )
    ds2 = load_dataset(
        "nickmuchi/financial-classification",
        split="train"
    )

    # Jeu 1 : 'tweet' → 'text', label déjà nommé 'label'
    if "tweet" in ds1.column_names:
        ds1 = ds1.rename_column("tweet", "text")

    # Jeu 2 : 'sentence' → 'text', 'labels' → 'label'
    if "sentence" in ds2.column_names:
        ds2 = ds2.rename_column("sentence", "text")
    if "labels" in ds2.column_names:
        ds2 = ds2.rename_column("labels", "label")


    print("🔹 Création des splits train/test…")
    ds1_split = ds1.train_test_split(test_size=test_size, seed=seed)
    ds2_split = ds2.train_test_split(test_size=test_size, seed=seed)

    train_ds = concatenate_datasets([ds1_split["train"], ds2_split["train"]])
    test_ds  = concatenate_datasets([ds1_split["test"],  ds2_split["test"]])

    final_dataset = DatasetDict({"train": train_ds, "test": test_ds})

    return final_dataset

def compute_metrics(pred):
    """
    Calcule les métriques d'évaluation pour les prédictions du modèle
"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_model(model_name, dataset, batch_size=16, num_epochs=3):
    """
    Fine-tune un modèle BERT sur le dataset fourni
    """
    print(f"Préparation du modèle: {model_name}")

    # Chargement du tokenizer et du modèle
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,  # 3 classes: Négatif, Neutre, Positif
    )

    text_column = None

    train_columns = dataset["train"].column_names

    if "text" in train_columns:
        text_column = "text"
    elif "sentence" in train_columns:
        text_column = "sentence"
    else:
        for col in train_columns:
            if "text" in col.lower() or "sentence" in col.lower() or "tweet" in col.lower() or "content" in col.lower():
                text_column = col
                break

    if text_column is None:
        raise ValueError("Impossible de déterminer la colonne contenant le texte dans le dataset")

    print(f"Utilisation de la colonne '{text_column}' comme source de texte")

    def tokenize_function(examples):
        """Tokenize les textes du dataset"""
        return tokenizer(
            examples[text_column],
            padding="max_length",
            truncation=True,
            max_length=128
        )

    print("Tokenisation du dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        desc="Tokenisation en cours"
    )

    tokenized_train = tokenized_dataset["train"]
    tokenized_test = tokenized_dataset["test"]

    tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    training_args = make_training_args(
        output_dir=f"./{model_name.split('/')[-1]}_results",
        batch_size=batch_size,
        num_epochs=num_epochs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics
    )

    print(f"Début de l'entraînement du modèle {model_name}")
    trainer.train()
    print("Évaluation du modèle")
    eval_results = trainer.evaluate()
    print(f"Résultats d'évaluation: {eval_results}")
    model_save_path = f"./{model_name.split('/')[-1]}_finetuned"
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Modèle sauvegardé dans: {model_save_path}")

    return tokenizer, model, trainer

def evaluate_detailed(model, tokenizer, dataset, text_column):
    model.eval()
    model.to(device)

    test_texts = dataset["test"][text_column]
    test_labels = dataset["test"]["label"]

    predictions = []

    for i in range(0, len(test_texts), 16):
        batch_texts = test_texts[i:i+16]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        predictions.extend(preds)

    # Rapport de classification
    target_names = ["Négatif", "Neutre", "Positif"]
    report = classification_report(test_labels, predictions, target_names=target_names)

    print("Rapport de classification détaillé:")
    print(report)

if __name__ == "__main__":
    dataset = load_and_prepare_datasets()
    text_column = None
    train_columns = dataset["train"].column_names

    if "text" in train_columns:
        text_column = "text"
    elif "sentence" in train_columns:
        text_column = "sentence"
    else:
        for col in train_columns:
            if "text" in col.lower() or "sentence" in col.lower() or "tweet" in col.lower() or "content" in col.lower():
                text_column = col
                break

    if text_column is None:
        raise ValueError("Impossible de déterminer la colonne contenant le texte dans le dataset")

    print(f"Utilisation de la colonne '{text_column}' pour le texte")

    # 2. Finetuning du modèle BERT
    bert_tokenizer, bert_model, bert_trainer = train_model(
        "bert-base-uncased",
        dataset,
        batch_size=16,
        num_epochs=3
    )

    # 3. Finetuning du modèle FinBERT
    finbert_tokenizer, finbert_model, finbert_trainer = train_model(
        "yiyanghkust/finbert-tone",  # FinBERT pré-entraîné sur des données financières
        dataset,
        batch_size=16,
        num_epochs=3
    )

    # 4. Évaluation détaillée des modèles
    print("\n--- Évaluation détaillée du modèle BERT ---")
    evaluate_detailed(bert_model, bert_tokenizer, dataset, text_column)

    print("\n--- Évaluation détaillée du modèle FinBERT ---")
    evaluate_detailed(finbert_model, finbert_tokenizer, dataset, text_column)

    # 5. Comparaison des performances
    print("\n--- Comparaison des performances ---")
    bert_eval = bert_trainer.evaluate()
    finbert_eval = finbert_trainer.evaluate()

    print("Performances BERT:")
    for metric, value in bert_eval.items():
        print(f"{metric}: {value:.4f}")

    print("\nPerformances FinBERT:")
    for metric, value in finbert_eval.items():
        print(f"{metric}: {value:.4f}")


