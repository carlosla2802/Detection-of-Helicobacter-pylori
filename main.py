from prep_data import prep_data_main
from autoencoder import Autoencoder
from train import train_autoencoder, make_losses_plot

import torch.nn as nn
import torch

def main(plot=True):

    # Definir rutas a los archivos de metadatos y directorio de imágenes
    annotated_patches_path = "window_metadata.csv"
    labeled_patients_path = "metadata.csv"
    annotated_images_dir = 'AnnotatedPatches'

    # Preparar data
    train_loader, val_loader = prep_data_main(annotated_patches_path, labeled_patients_path, annotated_images_dir)

    # Crear el modelo
    autoencoder = Autoencoder()

    # Definir la función de pérdida y el optimizador
    criterion = nn.MSELoss(); optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

    # Entrenar el modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); autoencoder.to(device)
    num_epochs = 100  # Ajustar num epochs
    
    history = train_autoencoder(
        autoencoder, criterion, optimizer, train_loader, val_loader, num_epochs, device
        )
    
    make_losses_plot(history)
    


if __name__ == '__main__':
    main(plot=True)