import os
import pandas as pd
from conllu import parse_incr

def process_ud_data(data_dir):
    stats = []
    
    for filename in os.listdir(data_dir):
        if filename.endswith(".conllu"):
            input_path = os.path.join(data_dir, filename)
            output_path = input_path.replace(".conllu", ".txt")
            
            sentence_count = 0
            token_count = 0
            
            with open(input_path, "r", encoding="utf-8") as f_in, \
                 open(output_path, "w", encoding="utf-8") as f_out:
                
                for tokenlist in parse_incr(f_in):
                    # Extract tokens and join them into a sentence
                    sentence = " ".join([token["form"] for token in tokenlist if isinstance(token["id"], int)])
                    f_out.write(sentence + "\n")
                    
                    sentence_count += 1
                    token_count += len(tokenlist)
            
            stats.append({
                "File": filename,
                "Language": filename.split("_")[0],
                "Split": filename.split("-")[-1].split(".")[0],
                "Sentences": sentence_count,
                "Tokens": token_count,
                "Output_File": os.path.basename(output_path)
            })
            print(f"✅ Processed: {filename} -> {os.path.basename(output_path)}")

    # Create Summary DataFrame
    df = pd.DataFrame(stats)
    df.to_csv("data_summary.csv", index=False)
    print("\n--- DATASET OVERVIEW ---")
    print(df.to_string(index=False))
    print("\nSummary saved to data_summary.csv")

if __name__ == "__main__":
    process_ud_data("data")
