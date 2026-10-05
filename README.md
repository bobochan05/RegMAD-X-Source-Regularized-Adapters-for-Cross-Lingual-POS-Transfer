# RegMAD-X — Source-Regularized Adapters for Cross-Lingual POS Transfer

RegMAD-X is an experimental implementation of source-informed L2 regularization for **MAD-X-style language adapters**. It investigates whether keeping a target-language adapter close to an English adapter during language-adaptive fine-tuning (LAFT) improves **English-to-target zero-shot Universal POS tagging**.

The experiments use `bert-base-multilingual-cased`, the Hugging Face `adapters` library, and Universal Dependencies (UD) treebanks. Baseline adapters are trained with masked-language modeling (MLM); regularized adapters add a penalty that aligns corresponding trainable adapter weights with the English source adapter.

## What is in this repository

- Baseline and regularized MLM/LAFT training scripts.
- English-to-Hindi, French, Arabic, and German zero-shot POS-tagging evaluations.
- A Hindi regularization-strength sweep.
- Utilities for converting UD CoNLL-U files to line-delimited text for MLM.
- The experiment's original data summary and dataset-distribution figure.

This is research code rather than a packaged training framework. Dataset files, trained adapters, checkpoints, and logs are intentionally excluded from version control.

## Method

MAD-X composes a language adapter with a task adapter. Here, the English language adapter is used while training the POS task adapter, then swapped for a target-language adapter at evaluation time.

For regularized target-language LAFT, `run_mlm_reg.py` optimizes the standard MLM objective plus the squared L2 distance between matching trainable target and English adapter parameters:

$$
\mathcal{L} = \mathcal{L}_{\mathrm{MLM}} + \lambda \sum_{p \in \phi_{\mathrm{target}}} \lVert p - p_{\mathrm{en}} \rVert_2^2.
$$

The regularizer expects the loaded source adapter to be named `en`; this is how the provided launch scripts and implementation are configured.

## Reported results

The table below records the results supplied with the original experiment. Training outputs and logs are not included, so treat these as reported results rather than independently verified benchmarks from this checkout.

| Target language | Transfer direction | Baseline F1 | Regularized F1 | Gain (F1 points) |
| --- | --- | ---: | ---: | ---: |
| Hindi | EN → HI | 0.5825 | **0.6229** | **+0.0404** |
| Arabic | EN → AR | 0.6239 | **0.6364** | **+0.0125** |
| German | EN → DE | 0.8699 | **0.8748** | +0.0049 |
| French | EN → FR | 0.8713 | 0.8713 | +0.0000 |

The reported Hindi sweep selected $\lambda = 0.001$. The standalone regularized training scripts currently use $\lambda = 0.1$, so set and record the value explicitly for any new experiment.

## Requirements

- Python 3.10 or later
- A CUDA-capable PyTorch installation for the supplied training configuration
- A Linux or WSL shell to run the `.sh` launchers

Clone the repository, create and activate an environment, install the CUDA build of PyTorch appropriate for your system, then install the Python dependencies:

```bash
git clone https://github.com/boboChan05/RegMAD-X-Source-Regularized-Adapters-for-Cross-Lingual-POS-Transfer.git
cd RegMAD-X-Source-Regularized-Adapters-for-Cross-Lingual-POS-Transfer

pip install adapters datasets evaluate conllu seqeval scikit-learn pandas matplotlib seaborn
```

`run_mlm.py` and `run_mlm_reg.py` require a Transformers version compatible with `adapters` and check for `transformers >= 4.44.0` at startup. Pin the exact dependency versions in your own environment before running a reproducibility study.

## Data setup

Download the relevant UD treebanks and place their CoNLL-U splits at the exact paths expected by the scripts:

```text
data/
├── en_ewt-ud-{train,dev,test}.conllu
├── hi_hdtb-ud-{train,dev,test}.conllu
├── fr_gsd-ud-{train,dev,test}.conllu
├── arabic/
│   └── ar_padt-ud-{train,dev,test}.conllu
└── german/
    └── de_gsd-ud-{train,dev,test}.conllu
```

The repository does not distribute any UD data. Make sure your use complies with each treebank's license.

Convert the MLM inputs after placing the data:

```bash
# Converts CoNLL-U files directly inside data/ (English, Hindi, and French)
python prepare_data.py

# Converts Arabic and German training splits and writes dataset_summary.csv
python convert.py

# Converts Arabic and German development splits
python convert_conllu.py
```

The expected text files are one sentence per line. `prepare_data.py` writes text beside every `.conllu` file in `data/`; the Arabic and German helper scripts cover their nested directories. The included `data_summary.csv` describes the original English, Hindi, and French preparation run.

## Experiment workflow

Run commands from the repository root. All output directories below are ignored by Git.

### 1. Train the source English language adapter

```bash
bash train_en.sh
```

This creates `output/adapter_en/mlm`, which is required by every target-language and POS evaluation script.

### 2. Train target language adapters

Baseline MLM/LAFT scripts:

```bash
bash train_hi.sh
bash train_fr.sh
bash run_base_ar.sh
bash run_base_de.sh
```

Regularized MLM/LAFT scripts:

```bash
bash run_reg_hi.sh
bash run_reg_fr.sh
bash train_ar.sh
bash train_de.sh
```

The default scripts train for 10 epochs with a Pfeiffer adapter configuration and write adapters under `output/`. Edit their batch sizes, epochs, data paths, GPU assignments, and `--reg_lambda` values for your hardware and experiment plan.

### 3. Train the English POS task adapter and evaluate zero-shot transfer

```bash
# Hindi and French
python train_task_eval.py       # baseline adapters
python train_task_eval_reg.py   # regularized adapters

# Arabic and German
python eval_ar_de_baseline.py
python eval_ar_de_reg.py
```

Each evaluation script trains a `pos_task` tagging adapter on English UD data, then evaluates it after composing the task adapter with each target-language adapter. It reports token-level accuracy and `seqeval` F1.

### 4. Sweep Hindi regularization strengths

```bash
bash sweep_lambda_hi.sh
```

The sweep evaluates $\lambda \in \{0.001, 0.01, 0.05, 0.1, 0.5\}$, appending F1 and accuracy to `lambda_sweep_results.csv`. The sweep uses five task-training epochs rather than the ten epochs used by the standard POS evaluation scripts.

## Repository map

| Path | Purpose |
| --- | --- |
| `run_mlm.py` | Baseline MLM/LAFT training adapted from the Hugging Face language-modeling example. |
| `run_mlm_reg.py` | MLM/LAFT training with `RegularizedAdapterTrainer` and `--reg_lambda`. |
| `train_{en,hi,fr}.sh` | Baseline English, Hindi, and French adapter training commands. |
| `run_base_{ar,de}.sh` | Baseline Arabic and German adapter training commands. |
| `run_reg_{hi,fr}.sh`, `train_{ar,de}.sh` | Regularized target-adapter training commands. |
| `train_task_eval*.py` | Baseline and regularized Hindi/French POS transfer experiments. |
| `eval_ar_de_*.py` | Baseline and regularized Arabic/German POS transfer experiments. |
| `sweep_lambda_hi.sh`, `eval_hi_sweep.py` | Hindi regularization sweep and evaluation. |
| `prepare_data.py`, `convert.py`, `convert_conllu.py` | UD CoNLL-U-to-text preparation utilities. |
| `plot_data.py`, `dataset_distribution.png` | Dataset-size visualization utility and generated figure. |
| `run_all_tasks.sh`, `launch_ar_de.sh` | Convenience GPU launchers. |

## Operational notes

- Several Python and shell scripts hard-code `CUDA_VISIBLE_DEVICES` values (`5`, `6`, or `7`). Change them before running on another machine. A Python assignment can override a value exported by the calling shell.
- `run_all_tasks.sh` and `launch_ar_de.sh` use `nohup`, background processes, and `fuser -k /dev/nvidia*`. Review and modify them before use: `fuser -k` terminates processes using the named GPU devices.
- The adapter, task, sweep, dataset, and log output paths are hard-coded for the original layout. Keep the stated directory structure or update the paths consistently.
- No trained checkpoints, raw datasets, environment lockfile, automated tests, or license file are currently included.

## Citation

If you build on this code, cite the MAD-X paper as well as the Universal Dependencies treebanks used in your experiment. Add a project-specific citation here if this work is released as a paper or report.

