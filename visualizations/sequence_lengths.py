from .utils import common_labels, load_labels
import pandas as pd
import matplotlib.pyplot as plt
from Bio import SeqIO

def run_all():
    df = load_labels()
    fasta_file = "scop_sf_represeq_lib_latest.fa.txt"
    seq_records = list(SeqIO.parse(fasta_file, "fasta"))

    assert len(seq_records) == len(df), "Number of FASTA sequences != number of rows in CSV!"

    # Extract protein lengths (index aligns with df rows)
    seq_lengths = [len(record.seq) for record in seq_records]

    # ---------------------------------------------------------------------
    # 4. For each label type in `common_labels`, plot 4 bars (one per label ID).
    #    Each bar shows the mean length of sequences that match that label ID.
    # ---------------------------------------------------------------------
    for label_type, label_ids in common_labels.items():
        mean_lengths = []

        # Compute mean length for each of the 4 IDs in this label type
        for label_id in label_ids:
            # Find rows in df where df[label_type] == label_id
            row_indices = df.index[df[label_type] == label_id]

            # Extract the lengths for these rows
            selected_lengths = [seq_lengths[i] for i in row_indices]

            # Compute mean length (handle case of no valid rows)
            if len(selected_lengths) > 0:
                mean_length = sum(selected_lengths) / len(selected_lengths)
            else:
                mean_length = 0

            mean_lengths.append(mean_length)

        # -----------------------------------------------------------------
        # 5. Plot a bar chart for this label type
        #    X-axis: label_ids, Y-axis: mean lengths
        # -----------------------------------------------------------------
        plt.figure(figsize=(6, 4))
        plt.bar(
            [str(x) for x in label_ids],  # convert IDs to strings for x-ticks
            mean_lengths,
            color="skyblue"
        )

        plt.title(f"Mean Protein Lengths for {label_type}")
        plt.xlabel("Label IDs")
        plt.ylabel("Mean Protein Length")

        # Adjust y-limits to give some headroom
        max_val = max(mean_lengths) if mean_lengths else 0
        plt.ylim([0, max_val * 1.2 if max_val > 0 else 1])

        plt.tight_layout()
        plt.savefig(f"results/statistics/seq_lengths_{label_type}.png")
        plt.close()  # Close the figure to avoid displaying it
