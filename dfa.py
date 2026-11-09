from collections import deque
from typing import Dict, Set, Tuple, Any, List
import itertools

"""
A simple customizable DFA implementation.
Accepts a set of all states, an alphabet, a transition function, a start state, and a set of accept states.
Allows classification of words and evaluation of error against a ground truth.
"""
class DFA:
    def __init__(self, states: Set[Any], alphabet: Set[str], transitions: Dict[Tuple[Any, str], Any], start_state: Any, accept_states: Set[Any]):
        self.states = set(states)
        self.alphabet = set(alphabet)
        self.transitions = transitions
        self.start_state = start_state
        self.accept_states = set(accept_states)

    # Determines if an input string is accepted by the DFA
    def classify(self, word: str) -> bool:
        curr = self.start_state
        for ch in word:
            curr = self.transitions.get((curr, ch))
            if curr is None:
                return False
        return curr in self.accept_states

    # Returns the state reached after processing the input string
    def get_state(self, word: str) -> Any:
        curr = self.start_state
        for ch in word:
            curr = self.transitions.get((curr, ch))
        return curr

    # Returns a dictionary mapping each state to the shortest prefix that reaches it from the start state
    def get_reachability_prefixes(self) -> Dict[Any, str]:
        visited = {self.start_state: ""}
        queue = deque([self.start_state])

        # Perform BFS to find shortest prefixes for each reachable state
        while queue:
            q = queue.popleft()
            p = visited[q]
            for ch in sorted(self.alphabet):
                nxt = self.transitions.get((q, ch))
                if nxt is not None and nxt not in visited:
                    visited[nxt] = p + ch
                    queue.append(nxt)
        return visited

    # Find the error rate of using a simplified DFA as compared to the ground truth
    def evaluate_error(self, test_words: List[str], ground_truth: List[bool]) -> float:
        if not test_words:
            return 0.0
        mismatches = sum(1 for w, truth in zip(test_words, ground_truth) if self.classify(w) != truth)
        return mismatches / len(test_words)

    def print(self) -> None:
        print("States:", self.states)
        print("Alphabet:", self.alphabet)
        print("Start State:", self.start_state)
        print("Accept States:", self.accept_states)
        print("Transitions:")
        for (state, char), next_state in sorted(self.transitions.items()):
            print(f"  δ({state}, '{char}') = {next_state}")

# Builds exact minimal DFA testing whether the k-th symbol from end is '1'
def build_suffix_tracking_dfa(k: int = 4) -> DFA:
    alphabet = {"0", "1"}
    states = {"".join(p) for p in itertools.product(["0", "1"], repeat=k)}
    transitions = {}
    for s in states:
        for bit in ["0", "1"]:
            # Shift window left, append new bit
            transitions[(s, bit)] = s[1:] + bit

    start_state = "0" * k
    accept_states = {s for s in states if s[0] == "1"}
    return DFA(states, alphabet, transitions, start_state, accept_states)

# Builds exact minimal DFA accepting binary strings whose integer value is divisible by m
def build_modulo_dfa(m: int = 7) -> DFA:
    alphabet = {"0", "1"}
    states = set(range(m))
    transitions = {}
    
    for rem in states:
        for bit in ["0", "1"]:
            nxt = (2 * rem + int(bit)) % m # next_state = (2 * curr_state + bit) % m.
            transitions[(rem, bit)] = nxt
            
    start_state = 0
    accept_states = {0}
    return DFA(states, alphabet, transitions, start_state, accept_states)