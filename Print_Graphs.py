import json
import matplotlib.pyplot as plt


def load_history_from_json(file_path):
    """Load the history JSON file and return it as a dictionary."""
    with open(file_path, 'r') as f:
        history = json.load(f)
    return history


def plot_validation_accuracy(files, path, comparator):
    """Plot validation accuracy from multiple TensorFlow history JSON files."""
    plt.figure(figsize=(10, 6))  # Set up the plot size

    # Loop through each JSON file, load history, and plot the validation accuracy
    for file in files:
        history = load_history_from_json(path + file)
        # Get the validation accuracy, handle cases where the key might differ
        val_accuracy = history.get(comparator, None)
        if val_accuracy is not None:
            epochs = range(1, len(val_accuracy) + 1)
            plt.plot(epochs, val_accuracy, label=f'History from {file}')
        else:
            print(f"Warning: No '{comparator}' found in {path}")

    # Label the axes and add title
    plt.title(f'{comparator} of activation functions compared')
    plt.xlabel('Epochs')
    plt.ylabel(f'{comparator}')
    plt.legend()  # Show the legend to differentiate between the plots
    plt.grid(True)
    plt.show()


# Example usage: Provide the paths to the JSON files
path = "C:\\Users\\Arthur\\Desktop\\Old_Laptop_AI\\ImplementAI\\history_multiclass\\"

#history_files = ['history_15_1_gelu_kern_3_opt_adam.json', 'history_15_1_relu_kern_3_opt_adam.json',
#                 'history_15_1_selu_kern_3_opt_adam.json', 'history_15_1_tanh_kern_3_opt_adam.json']
#plot_validation_accuracy(history_files, path, 'val_accuracy')

history_files = ['history_15_1_selu_kern_3_opt_adam.json', 'history_15_1_selu_kern_5_opt_adam.json',
                 'history_15_1_selu_kern_7_opt_adam.json']
plot_validation_accuracy(history_files, path, 'val_accuracy')