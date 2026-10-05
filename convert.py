import os
import matplotlib.pyplot as plt
from conllu import parse_incr
import pandas as pd

def extract_and_stats(input_path, output_path, lang):
    sent_count = 0
    token_count = 0
    
    try:
        with open(input_path, "r", encoding="utf-8") as f_in, \
             open(output_path, "w", encoding="utf-8") as f_out:
            print(f"🔄 Processing {lang} from {input_path}...")
            for sent in parse_incr(f_in):
                # Count tokens and build sentence
                words = [token["form"] for token in sent if isinstance(token["id"], int)]
                token_count += len(words)
                sent_count += 1
                
                # Write to text file
                f_out.write(" ".join(words) + "\n")
                
                if sent_count % 1000 == 0:
                    print(f"  > Processed {sent_count} sentences...")
                    
    except KeyboardInterrupt:
        print(f"\n⚠️ Interrupted! Saving partial progress for {lang}...")
    
    return {"Language": lang, "Sentences": sent_count, "Tokens": token_count}

# --- Execution ---
langs = [
    ("arabic", "data/arabic/ar_padt-ud-train.conllu", "data/arabic/train_ar.txt"),
    ("german", "data/german/de_gsd-ud-train.conllu", "data/german/train_de.txt")
]

stats_list = []
for lang_name, inp, outp in langs:
    stats = extract_and_stats(inp, outp, lang_name)
    stats_list.append(stats)

# --- Save Summary & Visualize ---
df = pd.DataFrame(stats_list)
df.to_csv("dataset_summary.csv", index=False)
print("\n✅ Summary saved to dataset_summary.csv")

# Visualization
plt.figure(figsize=(10, 5))
plt.bar(df['Language'], df['Tokens'], color=['teal', 'orange'])
plt.title("Token Distribution across New Languages")
plt.ylabel("Total Tokens")
plt.savefig("dataset_distribution.png")
print("📊 Chart saved as dataset_distribution.png")