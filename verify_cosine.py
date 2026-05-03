import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

np.random.seed(0)
targets = np.random.dirichlet(np.ones(10), size=100)
preds = np.random.dirichlet(np.ones(10), size=100)

old = np.mean([cosine_similarity([targets[i]], [preds[i]])[0][0] for i in range(len(targets))])
new = np.mean(
    np.sum(targets * preds, axis=1) /
    (np.linalg.norm(targets, axis=1) * np.linalg.norm(preds, axis=1) + 1e-12)
)

diff = abs(old - new)
print(f"old={old:.12f} new={new:.12f} diff={diff:.12f}")
print('PASS' if diff < 1e-6 else 'FAIL')
