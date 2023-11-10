from prep_data import prep_data_main
from autoencoder import Autoencoder
from train import train_autoencoder, load_model, make_losses_plot
from eval import evaluate_model, visualize_reconstructions, save_model
from classify import detect_h_pylori_all_validation, calculate_metrics, evaluate_reconstructions

import torch.nn as nn
import torch

from datetime import datetime

def main(use_pretrained=False, pretrained_path=None, plot=True):

    # Definir rutas a los archivos de metadatos y directorio de imágenes
    annotated_patches_path = "metadata/window_metadata.csv"
    labeled_patients_path = "metadata/metadata.csv"
    annotated_images_dir = "AnnotatedPatches"
    labeled_images_dir = "CroppedPatches"

    # Preparar data
    non_infected_train_loader, non_infected_val_loader, validation_patches_data_tensor, validation_patches_labels = prep_data_main(annotated_patches_path, labeled_patients_path, annotated_images_dir, labeled_images_dir)

    # Crear o cargar el modelo
    if use_pretrained and pretrained_path:
        autoencoder = load_model(pretrained_path, Autoencoder, device)
        print("Modelo cargado desde:", pretrained_path)    
        
    else:
        autoencoder = Autoencoder()

        # Definir la función de pérdida y el optimizador
        criterion = nn.MSELoss(); optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

        # Entrenar el modelo
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); autoencoder.to(device)
        num_epochs = 10 # Ajustar num epochs

    annotated_images_dir = 'AnnotatedPatches'

    # Preparar data
    non_infected_train_loader, non_infected_val_loader, anomaly_loader = prep_data_main(annotated_patches_path, labeled_patients_path, annotated_images_dir)

    # Crear o cargar el modelo
    if use_pretrained and pretrained_path:

        # -- para MAC --
        # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        autoencoder = load_model(pretrained_path, Autoencoder, device)
        print("Modelo cargado desde:", pretrained_path)    
        
    else:
        autoencoder = Autoencoder()

        # Definir la función de pérdida y el optimizador
        criterion = nn.MSELoss(); optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

        # Entrenar el modelo
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); autoencoder.to(device)
        
        # -- para MAC --
        # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        # autoencoder.to(device)
        
        num_epochs = 100 # Ajustar num epochs
        
        history = train_autoencoder(
            autoencoder, criterion, optimizer, non_infected_train_loader, non_infected_val_loader, num_epochs, device
            )
    
        if plot:
            make_losses_plot(history)
    
        # Guardar modelo
        date = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_model(autoencoder, f'autoencoder_{date}.pth')

    # -- para MAC --
    # autoencoder.to(device)

    # Evaluar el modelo
    reconstructions = evaluate_model(autoencoder, non_infected_val_loader, device)

    if plot:
        # Visualizar rconstrucciones
        visualize_reconstructions(reconstructions)
    

if __name__ == '__main__':
    main(use_pretrained=True, pretrained_path="autoencoder.pth", plot=True)
