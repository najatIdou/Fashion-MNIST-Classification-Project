import matplotlib.pyplot as plt
import numpy as np


label_names = { # Mapping of labels to clothing names
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}


def plot_images(images, labels, predicted_labels=None):
    """ Plot images along with their labels and predictions """
    fig, axes = plt.subplots(1, 10, figsize=(15, 15))
    labels = np.array(labels) # Convert to NumPy array

    for i in range(10): # Display the first 10 images
        ax = axes[i]

        ax.imshow(images[i], cmap="gray") # Display grayscale images


        label = label_names[labels[i]] # Retrieve and display true and predicted clothing names

        if predicted_labels is not None:
            predicted_label = label_names[predicted_labels[i]]
            ax.set_title(f"True: {label}\nPred: {predicted_label}", fontsize=10, pad=10)
        else:
            ax.set_title(f"True: {label}", fontsize=10, pad=10)

        ax.axis('off') # Hide axis for clarity

    plt.subplots_adjust(wspace=0.5) # Add spacing between subplots
    plt.show()

