import os
from conllu import parse_incr

def conllu_to_text(input_path, output_path):
    print(f"🔄 Starting conversion: {input_path}")
    count = 0
    try:
        with open(input_path, "r", encoding="utf-8") as f_in, \
             open(output_path, "w", encoding="utf-8") as f_out:
            for sentence in parse_incr(f_in):
                words = [token["form"] for token in sentence if isinstance(token["id"], int)]
                if words:
                    f_out.write(" ".join(words) + "\n")
                
                count += 1
                if count % 1000 == 0:
                    print(f"  > Processed {count} sentences...")
                    
        print(f"✅ Finished: {output_path} ({count} sentences)")
    except Exception as e:
        print(f"❌ Error: {e}")

# RESUME ONLY FOR DEV FILES
files_to_process = [
    ("data/arabic/ar_padt-ud-dev.conllu", "data/arabic/dev_ar.txt"),
    ("data/german/de_gsd-ud-dev.conllu", "data/german/dev_de.txt")
]

if __name__ == "__main__":
    for inp, outp in files_to_process:
        if os.path.exists(inp):
            conllu_to_text(inp, outp)
        else:
            print(f"⚠️ Missing file: {inp}")