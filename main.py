import itertools
from dfa import build_suffix_tracking_dfa, build_modulo_dfa
from hankel import build_basis_hankel
from svd import compute_hankel_svd, approximate_dfa_svd
from outputs import *

# Build the set of all possible words over the given alphabet up to a specified length
def generate_test_strings(alphabet: list, length: int = 8) -> list:
    words = []
    for l in range(length + 1):
        for prod in itertools.product(alphabet, repeat=l):
            words.append("".join(prod))
    return words

def main() -> None:
    MODE = 1
    # 1: Suffix Tracking DFA (4th symbol from end is '1')
    # 2: Modulo DFA (divisible by 7)

    ensure_output_dir("./output")
    print("=== DFA Approximation via Hankel Matrix SVD ===")
    if MODE == 1:
        print("Example DFA: accept when the k-th symbol from the end is '1'")
    elif MODE == 2:
        print("Example DFA: accept when the binary string is divisible by m")

    print("\n[1/5] Generating minimal exact DFA...")
    exact_dfa = build_suffix_tracking_dfa(k=4) if MODE == 1 else build_modulo_dfa(m=7)
    exact_count = len(exact_dfa.states)
    alphabet = sorted(list(exact_dfa.alphabet))
    exact_dfa.print()

    print("\n[2/5] Constructing Hankel matrix...")
    H, prefixes, suffixes = build_basis_hankel(exact_dfa, max_suffix_len=3)
    print(f"      Hankel matrix dimensions: {H.shape[0]}x{H.shape[1]}")
    export_hankel_csv(H, prefixes, suffixes)

    print("\n[3/5] Computing singular value decomposition...")
    _, singular_values, _ = compute_hankel_svd(H)
    plot_singular_values(singular_values)
    print(f"      Singular values: {singular_values.round(3)}")

    print("\n[4/5] Evaluating test set against ground truth...")
    test_words = generate_test_strings(alphabet, length=8)
    ground_truth = [exact_dfa.classify(w) for w in test_words]
    print(f"      Evaluated word count: {len(test_words)}")

    print("\n[5/5] Testing various K values...")
    results = []
    target_k_values = [k for k in range(2, exact_count + 1)]

    # For each target rank k, approximate the DFA using SVD and evaluate its error rate against the ground truth
    for k in target_k_values:
        approx_dfa = approximate_dfa_svd(exact_dfa, H, prefixes, alphabet, target_rank=k)
        err = approx_dfa.evaluate_error(test_words, ground_truth)
        results.append({ "states": k, "exact_states": exact_count, "error_rate": err })
        print(f"      k={k:2d} states | Error: {err * 100:6.2f}%")

    plot_tradeoff_curve(results)
    export_csv_table(results)
    print("\nExecution complete. Outputs saved in output folder.")

if __name__ == "__main__":
    main()