from prep_data import prep_data_main
from autoencoder import Autoencoder
from train import train_autoencoder, load_model, make_losses_plot
from eval import evaluate_model, visualize_reconstructions, save_model
from classify import visualize_reconstructions2, detect_h_pylori_all_validation, calculate_metrics, evaluate_reconstructions, calculate_roc_curve_optimal_infected_windows_patient


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
    non_infected_train_loader, non_infected_val_loader, pacients_data_tensors, pacients_labels = prep_data_main(annotated_patches_path, labeled_patients_path, annotated_images_dir, labeled_images_dir)

    # Crear o cargar el modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if use_pretrained and pretrained_path:
        autoencoder = load_model(pretrained_path, Autoencoder, device)
        print("Modelo cargado desde:", pretrained_path)    
        
    else:
        autoencoder = Autoencoder()

        # Definir la función de pérdida y el optimizador
        criterion = nn.MSELoss(); optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

        # Entrenar el modelo
        autoencoder.to(device)
        num_epochs = 10 # Ajustar num epochs
        
        history = train_autoencoder(
            autoencoder, criterion, optimizer, non_infected_train_loader, non_infected_val_loader, num_epochs, device
            )
    
        if plot:
            make_losses_plot(history)
    
        # Guardar modelo
        date = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_model(autoencoder, f'autoencoder_{date}.pth')


    # Evaluar el modelo
    reconstructions = evaluate_model(autoencoder, non_infected_val_loader, device)

    if plot:
        # Visualizar rconstrucciones
        visualize_reconstructions(reconstructions)
    
    reconstructed_pacients_for_visualization = evaluate_reconstructions(autoencoder, pacients_data_tensors, device)
    reconstructed_pacients_data_tensor = {pacient: [pair[1] for pair in pairs] for pacient, pairs in reconstructed_pacients_for_visualization.items()}


    if plot:
        visualize_reconstructions2(reconstructed_pacients_for_visualization)


    n_values = [0,1,2,3,4,5]
    fpr, tpr, thresholds, best_n = calculate_roc_curve_optimal_infected_windows_patient(pacients_labels, pacients_data_tensors, reconstructed_pacients_data_tensor, n_values, plot_img=False)

    pred_labels = detect_h_pylori_all_validation(pacients_data_tensors, reconstructed_pacients_data_tensor, n=best_n)

    report = calculate_metrics(pacients_labels, pred_labels)

    print("Metrics of the classification method: \n", report)


if __name__ == '__main__':
    main(use_pretrained=True, pretrained_path="autoencoder.pth", plot=True)