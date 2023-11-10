import torch
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(model, dataloader, device):
    model.eval()
    reconstructions = []
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs = inputs.to(device)
            reconstructed = model(inputs)
            reconstructions.append((inputs.cpu(), reconstructed.cpu()))
    return reconstructions

def visualize_reconstructions(reconstructions, num_images=5):
    # Asegúrate de que num_images no exceda el número de elementos en reconstructions
    num_images = min(num_images, len(reconstructions))
    
    fig, axes = plt.subplots(nrows=num_images, ncols=2, figsize=(10, 2 * num_images))
    
    for i in range(num_images):
        # Desempaquetar la tupla
        original_batch, reconstructed_batch = reconstructions[i]
        
        # Indexar para obtener la primera imagen del batch
        original = original_batch[0]
        reconstructed = reconstructed_batch[0]

        # Verifica si el tensor tiene una dimensión extra y elimínala
        if original.dim() == 4:
            original = original.squeeze(0)
        if reconstructed.dim() == 4:
            reconstructed = reconstructed.squeeze(0)

        # Permutar los ejes para que el canal venga al final
        original = original.numpy().transpose(1, 2, 0)
        reconstructed = reconstructed.numpy().transpose(1, 2, 0)

        # Asegúrate de que los datos están en el rango [0, 1]
        original = np.clip(original, 0, 1)
        reconstructed = np.clip(reconstructed, 0, 1)

        # Visualización
        axes[i, 0].imshow(original)
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(reconstructed)
        axes[i, 1].set_title('Reconstructed')
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()


def save_model(model, path):
    torch.save(model.state_dict(), path)

    


# Use these functions after training
#reconstructions = evaluate_model(autoencoder, val_loader, device)
#visualize_reconstructions(reconstructions)
#save_model(autoencoder, 'autoencoder.pth')