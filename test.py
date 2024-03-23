import numpy as np


"""cm test"""
y_true = np.random.randint(0, 2, 20)
y_pred = np.random.randint(0, 2, 20)

n_labels = len(np.unique(y_true))

cm = np.zeros((n_labels, n_labels), dtype=np.int8)

for i, j in zip(y_true, y_pred):
    cm[i, j] += 1

print(cm)
cm = cm / cm.sum(axis=1)
print(cm)

precision = cm[1, 1]

print("Precision: ", precision)