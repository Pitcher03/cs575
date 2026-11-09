from typing import List, Tuple, Dict
import numpy as np
from sklearn.cluster import KMeans
from dfa import DFA

def compute_hankel_svd(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    U, s, Vt = np.linalg.svd(H, full_matrices=False)
    return U, s, Vt

def approximate_dfa_svd(exact_dfa: DFA, H: np.ndarray, prefixes: List[str], alphabet: List[str], target_rank: int, seed: int = 42) -> DFA:
    target_k = min(target_rank, H.shape[0], H.shape[1])
    U, s, _ = np.linalg.svd(H, full_matrices=False)
    
    # Scaled low-dimensional coordinates for state equivalence
    embeddings = U[:, :target_k] @ np.diag(np.sqrt(s[:target_k]))
    kmeans = KMeans(n_clusters=target_k, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    # Map exact DFA state -> approximate cluster
    state_to_cluster = {}
    for p, lbl in zip(prefixes, labels):
        exact_st = exact_dfa.get_state(p)
        state_to_cluster[exact_st] = int(lbl)
    
    transitions: Dict[Tuple[int, str], int] = {}
    states = set(range(target_k))

    for cluster_id in states:
        for char in alphabet:
            cluster_members = [st for st, c_id in state_to_cluster.items() if c_id == cluster_id]
            destinations = []
            for st in cluster_members:
                nxt_exact = exact_dfa.transitions.get((st, char))
                if nxt_exact in state_to_cluster:
                    destinations.append(state_to_cluster[nxt_exact])
            if destinations:
                transitions[(cluster_id, char)] = max(set(destinations), key=destinations.count)
            else:
                transitions[(cluster_id, char)] = cluster_id

    start_state = state_to_cluster.get(exact_dfa.start_state, 0)
    accept_states = set()
    for cluster_id in states:
        votes = [H[i, 0] for i, lbl in enumerate(labels) if lbl == cluster_id]
        if votes and (sum(votes) / len(votes)) >= 0.5:
            accept_states.add(cluster_id)
            
    return DFA(states, set(alphabet), transitions, start_state, accept_states)