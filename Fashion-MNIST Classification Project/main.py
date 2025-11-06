from data_loading import load_data
from models import train_resnet50
from evaluation import evaluate_model
from visualization import plot_images

# Load and preprocess dataset
X_train_scaled, X_test_scaled, y_train, y_test, _, _ = load_data()

# Train ResNet-50 model and get predictions
resnet_model, y_pred_resnet_labels = train_resnet50(X_train_scaled, y_train, X_test_scaled, y_test)

# Evaluate ResNet-50 model
evaluate_model(y_test, y_pred_resnet_labels, "ResNet-50")

# Visualize predictions using original grayscale test images
original_images = X_test_scaled[:, :, :, 0].reshape(-1, 28, 28)
plot_images(original_images, y_test, predicted_labels=y_pred_resnet_labels)