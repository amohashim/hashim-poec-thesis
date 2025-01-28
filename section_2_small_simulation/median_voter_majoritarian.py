import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from collections import Counter

# Generate data for a bimodal distribution
np.random.seed(42)

# Parameters for the peaks
large_peak_mean = 10
large_peak_std = 2
large_peak_size = 1000

small_peak_mean = 20
small_peak_std = 1
small_peak_size = 200

# Generate the data
large_peak = np.random.normal(large_peak_mean, large_peak_std, large_peak_size)
small_peak = np.random.normal(small_peak_mean, small_peak_std, small_peak_size)

# Combine the peaks to form a bimodal distribution
bimodal_data = np.concatenate([large_peak, small_peak])

# Calculate the medians
median = np.median(bimodal_data)
large_peak_median = np.median(large_peak)
small_peak_median = np.median(small_peak)

# Fit a Gaussian Mixture Model (GMM) to the data
gmm = GaussianMixture(n_components=2, random_state=42)
bimodal_data_reshaped = bimodal_data.reshape(-1, 1)
gmm.fit(bimodal_data_reshaped)

# Generate samples from the Gaussian Mixture
gmm_samples, _ = gmm.sample(10_000)  # Sample 10,000 points
gmm_samples = gmm_samples.flatten()

# Generate data from a standard Gaussian distribution
standard_gaussian_samples = np.random.normal(0, 1, 10_000)  # Standard normal distribution (mean=0, std=1)

# Function to simulate elections
def simulate_elections(num_simulations, voter_sample_size, data):
    winning_candidates = []
    for _ in range(num_simulations):
        # Sample voters
        sampled_voters = np.random.choice(data, voter_sample_size, replace=True)
        
        # Select two candidates randomly
        candidates = np.random.choice(sampled_voters, 2, replace=False)
        
        # Calculate votes using vectorized operations
        distances = np.abs(sampled_voters[:, None] - candidates)
        votes = np.argmin(distances, axis=1)
        winner_index = Counter(votes).most_common(1)[0][0]
        
        # Record the winning candidate
        winning_candidates.append(candidates[winner_index])
    return winning_candidates

# Simulation parameters
num_simulations = 10_000
voter_sample_size = 1000

# Run simulations for the Gaussian Mixture samples
winning_candidates_gmm = simulate_elections(num_simulations, voter_sample_size, gmm_samples)
average_winner_position_gmm = np.mean(winning_candidates_gmm)

# Run simulations for the standard Gaussian samples
winning_candidates_standard = simulate_elections(num_simulations, voter_sample_size, standard_gaussian_samples)
average_winner_position_standard = np.mean(winning_candidates_standard)

# Create a figure with two subplots side by side
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Plot 1: Gaussian Mixture distribution
axes[0].hist(gmm_samples, bins=30, alpha=0.5, color='lightgray', edgecolor='black', density=True)
axes[0].axvline(median, color='darkred', linestyle='--', linewidth=2, label=f"Overall Median: {median:.2f}")
axes[0].axvline(large_peak_median, color='darkgreen', linestyle='--', linewidth=2, label=f"Large Peak Median: {large_peak_median:.2f}")
axes[0].axvline(small_peak_median, color='darkorange', linestyle='--', linewidth=2, label=f"Small Peak Median: {small_peak_median:.2f}")
axes[0].axvline(average_winner_position_gmm, color='purple', linestyle='-', linewidth=2.5, label=f"Avg Winner: {average_winner_position_gmm:.2f}")
axes[0].set_title("Gaussian Mixture Distribution", fontsize=16)
axes[0].axis("off")  # Turn off x and y-axis values

# Plot 2: Standard Gaussian distribution
axes[1].hist(standard_gaussian_samples, bins=30, alpha=0.5, color='lightgray', edgecolor='black', density=True)
axes[1].axvline(average_winner_position_standard, color='purple', linestyle='-', linewidth=2.5, label=f"Avg Winner: {average_winner_position_standard:.2f}")
axes[1].set_title("Standard Gaussian Distribution", fontsize=16)
axes[1].axis("off")  # Turn off x and y-axis values

# Add legends to each subplot
axes[0].legend(fontsize=12, loc="upper right")
axes[1].legend(fontsize=12, loc="upper right")

# Tight layout and display
plt.tight_layout()
plt.show()
