import numpy as np

# read loss values in .npy file and plot them
import matplotlib.pyplot as plt
def plot_loss(file_path):
    loss_values = np.load(file_path)
    print(loss_values)
    plt.plot(loss_values)
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid()
    plt.savefig('training_loss.png')
    plt.close()
    
    
if __name__ == "__main__":
    loss_path = "checkpoints/training_loss.npy"
    plot_loss(loss_path)
    print(f"Loss plot saved as 'training_loss.png'")
    