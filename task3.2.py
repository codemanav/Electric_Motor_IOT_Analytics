import numpy as np
from hmmlearn import hmm

# Simulated observations from Task 3.1 (must be a column vector)
observations = np.array(observations).reshape(-1, 1)  # 1000-length sequence from earlier

# Initialize and train HMM (Multinomial for discrete symbols)
model = hmm.MultinomialHMM(n_components=4, random_state=200600147, n_iter=100)
model.fit(observations)

# Extract trained parameters
P_learned = model.transmat_          # Learned transition matrix
B_learned = model.emissionprob_      # Learned emission matrix
pi_learned = model.startprob_        # Learned initial distribution

# Display results
print("Learned Transition Matrix (P):\n", P_learned)
print("\nLearned Emission Matrix (B):\n", B_learned)
print("\nLearned Initial Distribution (π):\n", pi_learned)
