import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def train_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    running_loss = 0.0
    for data in dataloader:
        inputs, _ = data  # Los autoencoders solo necesitan las entradas, no las etiquetas
    
        inputs = inputs.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, inputs)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    
    return epoch_loss

def validate_epoch(model, dataloader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for data in dataloader:
            inputs, _ = data
            inputs = inputs.to(device)
            
            outputs = model(inputs)
            loss = loss_fn(outputs, inputs)
            
            running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    
    return epoch_loss


def make_losses_plot(history):
    # Plot training & validation loss values
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Progression')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

#MAIN 
def train_autoencoder(model, criterion, optimizer, train_loader, val_loader, num_epochs, device):
    """
    Función para entrenar un autoencoder.

    Parámetros:
    - model: Modelo de autoencoder a entrenar.
    - criterion: Función de pérdida a utilizar.
    - optimizer: Optimizador para actualizar los pesos.
    - train_loader: DataLoader con los datos de entrenamiento.
    - val_loader: DataLoader con los datos de validación.
    - num_epochs: Número de épocas para entrenar.
    - device: Dispositivo en el que se ejecuta el entrenamiento ('cuda' o 'cpu').

    Retorna:
    - Un diccionario con el historial de pérdida de entrenamiento y validación.
    """

    # Mover el modelo al dispositivo correspondiente
    model.to(device)
    
    # Diccionario para almacenar el historial de pérdida
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(num_epochs):
        # Entrenamiento
        model.train()
        running_loss = 0.0
        for inputs, _ in train_loader:
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        
        # Validación
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for inputs, _ in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                running_loss += loss.item() * inputs.size(0)
        val_loss = running_loss / len(val_loader.dataset)
        
        # Guardar las métricas en el historial
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Imprimir las métricas
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return history
