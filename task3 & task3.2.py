import numpy as np
import numpy as np
from hmmlearn.hmm import CategoricalHMM
# Set random seed using student ID
np.random.seed(200600147)

# Parameters
N = 4  # number of states
M = 3  # number of observation symbols
T = 1000  # sequence length

# Transition matrix P (NxN)
P = np.random.rand(N, N)
P = P / P.sum(axis=1, keepdims=True)  # Normalize rows

# Emission matrix B (NxM)
B = np.random.rand(N, M)
B = B / B.sum(axis=1, keepdims=True)

# Initial state distribution π
pi = np.zeros(N)
pi[0] = 1.0  # Start always from state 0

# Generate state and observation sequences
states = [0]
observations = [np.random.choice(M, p=B[0])]  # first observation

for _ in range(1, T):
    current_state = states[-1]
    next_state = np.random.choice(N, p=P[current_state])
    emitted_symbol = np.random.choice(M, p=B[next_state])
    states.append(next_state)
    observations.append(emitted_symbol)

# Output (for inspection)
print("Initial State Distribution (π):", pi)
print("Transition Matrix (P):\n", P)
print("Emission Matrix (B):\n", B)
print("First 10 Hidden States (Q):", states[:10])
print("First 10 Observations (O):", observations[:10])

# Convert to numpy array and reshape
obs_seq = np.array(observations).reshape(-1, 1)  # Shape: (T, 1)

# Train a 4-state categorical HMM on the sequence
model = CategoricalHMM(n_components=4, n_iter=100, random_state=200600147)
model.fit(obs_seq)

# Extract learned parameters
P_learned = model.transmat_
B_learned = model.emissionprob_
pi_learned = model.startprob_

# Display results
print("Learned Transition Matrix (P):\n", P_learned)
print("\nLearned Emission Matrix (B):\n", B_learned)
print("\nLearned Initial Distribution (π):\n", pi_learned)
