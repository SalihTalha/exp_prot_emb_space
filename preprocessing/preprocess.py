import pandas as pd
from Bio import SeqIO
from tqdm import tqdm


def search_and_retrieve(file_path, search_number):
    with open(file_path, 'r') as file:
        while True:
            # Read two lines at a time
            line = file.readline()
            if line.startswith("#"):
                continue

            # Check if the number exists in the first line
            if str(search_number) in line:
                return line

res = []

superfamilies = [i.id for i in SeqIO.parse("scop_sf_represeq_lib_latest.fa.txt", "fasta")]

for s in tqdm(superfamilies):
    entries = search_and_retrieve("scop-cla-latest.txt", s).split(" ")[-1][:-1].split(",")
    parsed_dict = {key: int(value) for key, value in (entry.split('=') for entry in entries)}

    res.append(parsed_dict)

print(res)
df = pd.DataFrame(res)
df.to_csv("labels_sf.csv")
