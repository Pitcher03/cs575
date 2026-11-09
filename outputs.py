import os
import csv
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import numpy as np

def ensure_output_dir(dir_path: str = "./output") -> None:
    os.makedirs(dir_path, exist_ok=True)

# Graphs the singular value spectrum of the Hankel matrix
def plot_singular_values(singular_values: np.ndarray, filepath: str = "./output/svd.png") -> None:
    plt.figure(figsize=(6, 4))
    ranks = np.arange(1, len(singular_values) + 1)
    plt.plot(ranks, singular_values, "o-", color="royalblue", lw=1.5, markersize=5)
    plt.title("Singular Value Weights")
    plt.xlabel(r"Index $i$")
    plt.ylabel(r"Singular Value $\sigma$")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()

# Visualize the tradeoff between state budget and classification error
def plot_tradeoff_curve(results: List[Dict[str, Any]], filepath: str = "./output/tradeoff.png") -> None:
    k_vals = [r["states"] for r in results]
    errors = [r["error_rate"] * 100 for r in results]
    plt.figure(figsize=(6, 4))
    plt.plot(k_vals, errors, "s-", color="blue", lw=1.5, markersize=5)
    plt.title("State Budget vs. Classification Error")
    plt.xlabel(r"Target State Count ($k$)")
    plt.ylabel("Classification Error Rate (%)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()

# Exports the results of the tradeoff analysis to a CSV file to add to the report
def export_csv_table(results: List[Dict[str, Any]], filepath: str = "./output/results.csv") -> None:
    fieldnames = ["Target States (k)", "Exact States", "Error Rate (%)", "State Reduction (%)"]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for r in results:
            reduction = (1.0 - (r["states"] / r["exact_states"])) * 100
            writer.writerow([r["states"], r["exact_states"], f"{r['error_rate'] * 100:.2f}%", f"{reduction:.1f}%"])

# Print the hankel matrix to a csv file for inclusion in the report
def export_hankel_csv(H: np.ndarray, prefixes: List[str], suffixes: List[str], filepath: str = "./output/hankel.csv") -> None:
    # Label empty string as epsilon for clarity in report tables
    col_headers = ["Prefix \\ Suffix"] + [s if s != "" else "ε" for s in suffixes]
    
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(col_headers)
        for i, prefix in enumerate(prefixes):
            p_label = prefix if prefix != "" else "ε"
            row_data = [int(val) for val in H[i]]
            writer.writerow([p_label] + row_data)