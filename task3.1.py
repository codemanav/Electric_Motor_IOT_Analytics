import numpy as np

# Define HMM parameters (assume these are already generated as in Task 3.1)
# P: transition matrix, B: emission matrix, pi: initial state distribution
# O_given: 0-indexed observation sequence
O_given = [0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2]
N = P.shape[0]
T = len(O_given)

# Initialize delta and psi tables
delta = np.zeros((T, N))
psi = np.zeros((T, N), dtype=int)

# Step 1: Initialization
delta[0] = pi * B[:, O_given[0]]

# Step 2: Recursion
for t in range(1, T):
    for j in range(N):
        max_prob = delta[t - 1] * P[:, j]
        psi[t, j] = np.argmax(max_prob)
        delta[t, j] = np.max(max_prob) * B[j, O_given[t]]

# Step 3: Termination
Q_star = np.zeros(T, dtype=int)
Q_star[-1] = np.argmax(delta[-1])

# Step 4: Path backtracking
for t in reversed(range(1, T)):
    Q_star[t - 1] = psi[t, Q_star[t]]

# Output the most probable state sequence
print("Most probable state sequence (Q*):", Q_star.tolist())