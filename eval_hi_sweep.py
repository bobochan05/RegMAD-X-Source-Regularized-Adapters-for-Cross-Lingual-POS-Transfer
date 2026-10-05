import os
import argparse
import numpy as np
import pandas as pd
from conllu import parse_incr
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoConfig, TrainingArguments, EvalPrediction, set_seed
from adapters import AutoAdapterModel, AdapterTrainer
from adapters.composition import Stack
from seqeval.metrics import f1_score
from sklearn.metrics import accuracy_score

# ===============================
# 1️⃣ SWEEP CONFIGURATION
# ===============================
parser = argparse.ArgumentParser()
parser.add_argument("--target_adapter_path", type=str, required=True, help="Path to the trained regularized Hindi adapter")
parser.add_argument("--lambda_val", type=str, required=True, help="Lambda value used for this run")
parser.add_argument("--seed", type=int, default=42)
args_sweep = parser.parse_args()

SEED = args_sweep.seed
set_seed(SEED)
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
MODEL_NAME = "bert-base-multilingual-cased"

LOCAL_EN_ADAPTER = "./output/adapter_en/mlm"
LOCAL_HI_ADAPTER = args_sweep.target_adapter_path # Loaded from sweep script

DATA_PATHS = {
    "train_en": "data/en_ewt-ud-train.conllu",
    "val_en": "data/en_ewt-ud-dev.conllu",
    "test_hi": "data/hi_hdtb-ud-test.conllu"
}

# ===============================
# 2️⃣ DATA UTILITIES
# ===============================
def load_conllu(path):
    tokens, pos_tags = [], []
    with open(path, "r", encoding="utf-8") as f:
        for sent in parse_incr(f):
            t, p = [], []
            for tok in sent:
                if isinstance(tok["id"], int):
                    t.append(tok["form"])
                    p.append(tok["upos"])
            tokens.append(t)
            pos_tags.append(p)
    return Dataset.from_dict({"tokens": tokens, "pos_tags": pos_tags})

dataset = DatasetDict({
    "train": load_conllu(DATA_PATHS["train_en"]),
    "validation": load_conllu(DATA_PATHS["val_en"]),
    "test_hi": load_conllu(DATA_PATHS["test_hi"])
})

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
label_list = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label2id.items()}

def tokenize_and_align(examples):
    tokenized = tokenizer(examples["tokens"], is_split_into_words=True, truncation=True, padding="max_length", max_length=128)
    labels_out = []
    for i, tags in enumerate(examples["pos_tags"]):
        word_ids = tokenized.word_ids(i)
        label_ids = []
        prev = None
        for w in word_ids:
            if w is None: label_ids.append(-100)
            elif w != prev: label_ids.append(label2id.get(tags[w], label2id["X"]))
            else: label_ids.append(-100)
            prev = w
        labels_out.append(label_ids)
    tokenized["labels"] = labels_out
    return tokenized

dataset = dataset.map(tokenize_and_align, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# ===============================
# 3️⃣ MODEL & ADAPTER INITIALIZATION
# ===============================
config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=len(label_list), id2label=id2label, label2id=label2id)
model = AutoAdapterModel.from_pretrained(MODEL_NAME, config=config)

model.load_adapter(LOCAL_EN_ADAPTER, load_as="en")
model.load_adapter(LOCAL_HI_ADAPTER, load_as="hi_target")

model.add_adapter("pos_task")
model.add_tagging_head("pos_task", num_labels=len(label_list))

model.train_adapter("pos_task")
model.active_adapters = Stack("en", "pos_task")

def compute_metrics(p: EvalPrediction):
    predictions = np.argmax(p.predictions, axis=-1)
    flat_preds, flat_labels = [], []
    true_preds_nested, true_labels_nested = [], []
    for pred_seq, label_seq in zip(predictions, p.label_ids):
        tp, tl = [], []
        for p_i, l_i in zip(pred_seq, label_seq):
            if l_i != -100:
                tp.append(id2label[p_i]); tl.append(id2label[l_i])
                flat_preds.append(p_i); flat_labels.append(l_i)
        true_preds_nested.append(tp); true_labels_nested.append(tl)
    return {"f1": f1_score(true_labels_nested, true_preds_nested), "accuracy": accuracy_score(flat_labels, flat_preds)}

# ===============================
# 4️⃣ TASK TRAINING & EVALUATION
# ===============================
trainer = AdapterTrainer(
    model=model,
    args=TrainingArguments(
        output_dir=f"./temp_sweep_{args_sweep.lambda_val}",
        learning_rate=1e-4,
        num_train_epochs=5, # Reduced for sweep efficiency
        per_device_train_batch_size=32,
        seed=SEED,
        data_seed=SEED,
        full_determinism=True,
        dataloader_num_workers=0,
        remove_unused_columns=False
    ),
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_metrics
)

print(f"🚀 Training Task Head for Lambda: {args_sweep.lambda_val}")
trainer.train()

# Swapping to the regularized Hindi adapter for Zero-Shot
model.active_adapters = Stack("hi_target", "pos_task")
results = trainer.evaluate(eval_dataset=dataset["test_hi"])

# ===============================
# 5️⃣ SAVE & VISUALIZE RESULTS
# ===============================
log_file = "lambda_sweep_results.csv"
new_data = {
    "Lambda": [args_sweep.lambda_val],
    "F1": [round(results["eval_f1"], 4)],
    "Accuracy": [round(results["eval_accuracy"], 4)]
}
new_df = pd.DataFrame(new_data)

# Append to cumulative CSV file
if not os.path.isfile(log_file):
    new_df.to_csv(log_file, index=False)
else:
    new_df.to_csv(log_file, mode='a', header=False, index=False)

# Visualize final table
print("\n" + "="*40)
print(f"🏁 RESULTS FOR LAMBDA: {args_sweep.lambda_val}")
summary_df = pd.read_csv(log_file)
print(summary_df.to_string(index=False))
print("="*40)