# MAD-X with $L_2$ Regularization for Cross-Lingual Transfer

This repository implements a regularized version of the MAD-X (Adapter-based) framework for Cross-Lingual Zero-Shot transfer. We introduce a weighted $L_2$ penalty between the source (English) and target language adapters to maintain structural alignment during Language Adaptive Fine-Tuning (LAFT).

## 🚀 Key Findings
- **High-Distance Boost:** We observed a significant improvement (~4% F1) in Zero-Shot POS tagging for linguistically distant languages like **Hindi** and **Arabic**.
- **Optimal Regularization:** Hyperparameter sweeps identified **$\lambda = 0.001$** as the optimal penalty strength.
- **Linguistic Pattern:** The effectiveness of regularization is inversely proportional to the linguistic similarity between the source and target language.

## 📊 Results Summary
| Language | Pair | Baseline F1 | Regularized F1 | Absolute Gain |
| :--- | :--- | :--- | :--- | :--- |
| Hindi (HI) | EN $\rightarrow$ HI | 0.5825 | **0.6229** | **+4.04%** |
| Arabic (AR) | EN $\rightarrow$ AR | 0.6239 | **0.6364** | **+1.25%** |
| German (DE) | EN $\rightarrow$ DE | 0.8699 | **0.8748** | +0.49% |
| French (FR) | EN $\rightarrow$ FR | 0.8713 | 0.8713 | 0.00% |

## 🛠️ Installation & Setup
1. **Clone the repository:**
   \`\`\`bash
   git clone https://github.com/Ivar1331/madx-real.git
   cd madx-real
   \`\`\`
2. **Install dependencies:**
   \`\`\`bash
   pip install transformers adapters datasets conllu scikit-learn seqeval
   \`\`\`

## 📂 Project Structure
- \`run_mlm_reg.py\`: Modified MLM training script with $L_2$ regularization logic.
- \`train_task_eval_reg.py\`: Task training (POS) and Zero-shot evaluation script.
- \`sweep_lambda_hi.sh\`: Bash script for hyperparameter search over $\lambda$ values.
- \`launch_ar_de.sh\`: Optimized parallel launcher for Arabic and German experiments.

## 🧪 Methodology
During the training of the target language adapter ($\phi_{target}$), we add the following penalty to the loss function:
$$\mathcal{L}_{total} = \mathcal{L}_{MLM} + \lambda \sum || \phi_{target} - \phi_{source} ||^2$$
This ensures that the target adapter does not drift too far from the source adapter's feature space, which is critical for Zero-Shot transfer.

