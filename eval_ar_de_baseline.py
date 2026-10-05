import os
import numpy as np
from conllu import parse_incr
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoConfig, TrainingArguments, EvalPrediction, set_seed
from adapters import AutoAdapterModel, AdapterConfig, AdapterTrainer
from adapters.composition import Stack
from seqeval.metrics import f1_score
from sklearn.metrics import accuracy_score

# Reproducibility Setup
SEED = 42
set_seed(SEED)

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
MODEL_NAME = "bert-base-multilingual-cased"

LOCAL_EN_ADAPTER = "./output/adapter_en/mlm"
LOCAL_AR_ADAPTER = "./output/adapter_ar_baseline/mlm"
LOCAL_DE_ADAPTER = "./output/adapter_de_baseline/mlm"


DATA_PATHS = {
    "train_en": "data/en_ewt-ud-train.conllu",
    "val_en": "data/en_ewt-ud-dev.conllu",
    "test_ar": "data/arabic/ar_padt-ud-test.conllu",
    "test_de": "data/german/de_gsd-ud-test.conllu"
}

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
    "test_ar": load_conllu(DATA_PATHS["test_ar"]),
    "test_de": load_conllu(DATA_PATHS["test_de"]),
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

config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=len(label_list), id2label=id2label, label2id=label2id)
model = AutoAdapterModel.from_pretrained(MODEL_NAME, config=config)

model.load_adapter(LOCAL_EN_ADAPTER, load_as="en")
model.load_adapter(LOCAL_AR_ADAPTER, load_as="ar")
model.load_adapter(LOCAL_DE_ADAPTER, load_as="de")

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

trainer = AdapterTrainer(
    model=model,
    args=TrainingArguments(
        output_dir="./task_output_baseline", 
        learning_rate=1e-4, 
        num_train_epochs=10, 
        per_device_train_batch_size=32, 
        overwrite_output_dir=True, 
        remove_unused_columns=False,
        seed=SEED,
        data_seed=SEED,
        full_determinism=True,
        dataloader_num_workers=0
    ),
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_metrics
)

trainer.train()

results = {}
for lang in ["ar", "de"]:
    model.active_adapters = Stack(lang, "pos_task")
    eval_trainer = AdapterTrainer(
        model=model,
        args=TrainingArguments(output_dir=f"./eval_{lang}_baseline", remove_unused_columns=False, dataloader_num_workers=0),
        eval_dataset=dataset[f"test_{lang}"],
        compute_metrics=compute_metrics,
    )
    results[lang] = eval_trainer.evaluate()

print("\n" + "="*50)
print(f"{'Language':<12} | {'F1-Score':<12} | {'Accuracy':<12}")
for lang, res in results.items():
    print(f"{lang.upper():<12} | {res['eval_f1']:<12.4f} | {res['eval_accuracy']:<12.4f}")
print("="*50)