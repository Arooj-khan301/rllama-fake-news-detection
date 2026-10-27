"""End-to-end training (4-bit QLoRA SFT) and held-out evaluation entry point."""

import argparse
import json
import os
import random
import numpy as np
import pandas as pd

from src.config import ExperimentConfig
from src.data_utils import make_stratified_splits, prepare_dataframe
from src.prompts import build_prompt, parse_label
from src.retrieval import ContextualRankRAGRetriever


def run_pipeline(cfg: ExperimentConfig) -> None:
    """Executes data preparation, contextual retrieval, QLoRA SFT, and evaluation."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
    )
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    raw_df = pd.read_csv(cfg.csv_path)
    df = prepare_dataframe(raw_df, text_col=cfg.text_col, label_col=cfg.label_col)
    train_df, val_df, test_df = make_stratified_splits(df, seed=cfg.seed)

    if cfg.max_train_rows is not None:
        train_df = train_df.iloc[: cfg.max_train_rows].reset_index(drop=True)

    retriever = ContextualRankRAGRetriever(
        embed_model_name=cfg.embed_model_name,
        reranker_name=cfg.reranker_name,
        top_candidates=cfg.top_candidates,
        top_k=cfg.top_k,
    )
    retriever.fit_index(
        train_df["text"].tolist(), train_df["label"].astype(int).tolist()
    )

    train_records = []
    for idx, row in train_df.iterrows():
        contexts = retriever.retrieve_and_rerank(row["text"], exclude_index=int(idx))
        prompt = build_prompt(row["text"], contexts)
        target = "Fake" if int(row["label"]) == 0 else "Real"
        train_records.append({
            "text": f"{prompt}\nLabel: {target}\nReason: Classification based on the labelled dataset example."
        })

    val_records = []
    for _, row in val_df.iterrows():
        contexts = retriever.retrieve_and_rerank(row["text"])
        prompt = build_prompt(row["text"], contexts)
        target = "Fake" if int(row["label"]) == 0 else "Real"
        val_records.append({
            "text": f"{prompt}\nLabel: {target}\nReason: Classification based on the labelled dataset example."
        })

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if torch.cuda.is_available():
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=quant_config,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    if torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    sft_args = SFTConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=cfg.learning_rate,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available(),
        bf16=False,
        max_length=cfg.max_length,
        packing=False,
        report_to="none",
        seed=cfg.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=Dataset.from_list(train_records),
        eval_dataset=Dataset.from_list(val_records),
        peft_config=lora_config,
        processing_class=tokenizer,
    )
    trainer.train()

    eval_df = (
        test_df
        if cfg.max_test_rows is None
        else test_df.iloc[: cfg.max_test_rows].copy()
    )
    prediction_rows = []
    model.eval()
    for _, row in eval_df.reset_index(drop=True).iterrows():
        contexts = retriever.retrieve_and_rerank(row["text"])
        prompt = build_prompt(row["text"], contexts)
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=cfg.max_length
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        continuation = output[0][inputs["input_ids"].shape[1] :]
        answer = tokenizer.decode(continuation, skip_special_tokens=True)
        predicted = parse_label(answer)
        prediction_rows.append({
            "text": row["text"],
            "true_label": int(row["label"]),
            "predicted_label": predicted,
            "generated_answer": answer,
        })

    os.makedirs(cfg.output_dir, exist_ok=True)
    pred_df = pd.DataFrame(prediction_rows)
    pred_df.to_csv(os.path.join(cfg.output_dir, "test_predictions.csv"), index=False)

    valid_df = pred_df.dropna(subset=["predicted_label"]).copy()
    coverage = len(valid_df) / max(1, len(pred_df))
    summary = {
        "base_model": cfg.base_model,
        "embedding_model": cfg.embed_model_name,
        "reranker_model": cfg.reranker_name,
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "test_rows": int(len(eval_df)),
        "prediction_coverage": float(coverage),
    }
    if len(valid_df) > 0:
        y_true = valid_df["true_label"].astype(int)
        y_pred = valid_df["predicted_label"].astype(int)
        summary["accuracy"] = float(accuracy_score(y_true, y_pred))
        summary["macro_f1"] = float(
            precision_recall_fscore_support(
                y_true, y_pred, average="macro", zero_division=0
            )[2]
        )
        print(classification_report(y_true, y_pred, target_names=["Fake", "Real"]))
        print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

    with open(os.path.join(cfg.output_dir, "experiment_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RLLaMA Fake News Detection")
    parser.add_argument("--csv-path", default="data/FakeNewsNet.csv")
    parser.add_argument("--text-col", default="title")
    parser.add_argument("--label-col", default="real")
    parser.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    args = parser.parse_args()
    config = ExperimentConfig(
        csv_path=args.csv_path,
        text_col=args.text_col,
        label_col=args.label_col,
        base_model=args.base_model,
    )
    run_pipeline(config)
