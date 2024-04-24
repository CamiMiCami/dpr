import numpy as np

# Define the choices
choices = [0, 1]

# Set the size of the numpy array
array_size = ((3,))  # For example, a 5x5 array

# Initialize the numpy array with random choices from the given options
y = np.random.choice(choices, size=array_size)
X = np.random.randint(1, 3, size=(3,3))

print(X)
print(y)

classes = np.unique(y)
n_classes = len(classes)
n_samples = len(y)


