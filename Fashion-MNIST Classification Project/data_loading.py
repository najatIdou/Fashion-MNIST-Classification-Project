import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from torchvision import transforms

def load_data():
    # Fetch Fashion-MNIST dataset
    fashion_mnist = fetch_openml('Fashion-MNIST', version=1)
    X, y = fashion_mnist.data, fashion_mnist.target
    y = y.astype(int)

    # Use a subset of the dataset for faster training (optional)
    X = X[:10000]
    y = y[:10000]

    # Reshape to 28x28 and normalize
    X = X.values.reshape(-1, 28, 28, 1).astype(np.float32) / 255.0

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Ensure labels are NumPy arrays
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Convert grayscale to 3 channels for ResNet input
    X_train = np.repeat(X_train, 3, axis=-1)
    X_test = np.repeat(X_test, 3, axis=-1)

    return X_train, X_test, y_train, y_test, None, None
