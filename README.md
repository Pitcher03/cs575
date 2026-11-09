# Approximate DFA Minimization via Hankel Low-Rank SVD

Empirical evaluation of deterministic finite automata (DFA) compression using truncated Singular Value Decomposition (SVD) on finite Hankel basis matrices.

## Overview
1. Constructs an exact minimal DFA for a parameterized target language.
2. Extracts a finite prefix-suffix Hankel incidence matrix $H \in \{0, 1\}^{|P| \times |S|}$.
3. Decomposes $H$ via SVD to inspect singular value spectra.
4. Truncates rank to $k < |Q_{\min}|$ and clusters projected row embeddings ($k$-means) to reconstruct approximate $k$-state DFAs.
5. Evaluates empirical classification error against exact language truth over uniform test sets.

## Usage
In main.py, a handful of constants can be tuned to alter the behavior. 
- Setting MODE=1 runs the program using a DFA that identifies strings where the fourth-to-last symbol from the end is '1'.
- Setting MODE=2 runs the program using a DFA that identifies strings that have a decimal value divisible by 7.
- Modifying the k parameter in mode 1 or the m parameter in mode 2 will slightly alter the DFA acceptance conditions.

## Installation & Execution
```bash
pip install -r requirements.txt
python main.py