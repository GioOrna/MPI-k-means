import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

"""## Generate data:"""

np.random.seed(5032024)
mu = [np.array([1, 1, 1]), np.array([5, 5, 5])] # means
sig = [np.array([[1, 0, 0], # cov matrixes
                 [0, 1, 0],
                 [0, 0, 1]]), np.array([[1, 0, 0],
                                        [0, 1, 0],
                                        [0, 0, 1]])]
data_per_cluster=[100, 50] # number of observations per cluster
output_filename = "output.csv"

X=0
for i in range(len(mu)):
  if i == 0:
    data = np.random.multivariate_normal(mu[i], sig[i], data_per_cluster[i])
  else:
    data = np.vstack([data, np.random.multivariate_normal(mu[i], sig[i], data_per_cluster[i])])

# Standardize the dataset
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

"""## Plot the dataset to check if it's ok:"""

# Plot results
n = len(mu[0])
fig, axes = plt.subplots(n, n, figsize=(12, 12))

for i in range(n):
  for j in range(n):
    if i != j:
     axes[i][j].scatter(data_scaled[:,i], data_scaled[:,j], color='black', marker='o') # plot the data
     axes[i][j].set_xlabel('Var '+str(i))
     axes[i][j].set_ylabel('Var '+str(j))

plt.tight_layout()
plt.show()

"""## Save the dataset:"""

# Save the dataset
rows, cols = data_scaled.shape
with open(output_filename, "w") as f:
    # write row/col info
    f.write(f"{rows},{cols}\n")
    np.savetxt(f, data_scaled, delimiter=",", fmt="%s")
