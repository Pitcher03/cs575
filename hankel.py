import itertools
from typing import List, Tuple
import numpy as np
from dfa import DFA

# Generates all words over the given alphabet up to the specified maximum length
def generate_bounded_words(alphabet: List[str], max_len: int) -> List[str]:
    words = [""]
    for l in range(1, max_len + 1):
        for prod in itertools.product(alphabet, repeat=l):
            words.append("".join(prod))
    return words

# Builds the Hankel matrix for a given DFA and a maximum suffix length
def build_basis_hankel(dfa: DFA, max_suffix_len: int = 4) -> Tuple[np.ndarray, List[str], List[str]]:
    # Prefixes = Reachability basis covering all reachable states
    state_to_prefix = dfa.get_reachability_prefixes()
    prefixes = [state_to_prefix[q] for q in sorted(list(dfa.states)) if q in state_to_prefix]
    
    # Suffixes = All words over alphabet up to max_suffix_len
    suffixes = generate_bounded_words(sorted(list(dfa.alphabet)), max_suffix_len)

    H = np.zeros((len(prefixes), len(suffixes)), dtype=float)
    # Iterate over all prefixes 
    for i, u in enumerate(prefixes):
        # Iterate over all suffixes
        for j, v in enumerate(suffixes):
            # Find the classification of each concatenation of prefix and suffix
            H[i, j] = 1.0 if dfa.classify(u + v) else 0.0
            
    return H, prefixes, suffixes