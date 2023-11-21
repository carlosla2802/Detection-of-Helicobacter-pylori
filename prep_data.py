import pandas as pd
from PIL import Image
import os
import numpy as np
import cv2

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

def load_metadata(path):
    """ Carga los metadatos de un CSV. """
    return pd.read_csv(path)


# Definir función para cargar y procesar una imagen
def load_and_process_image(image_path, target_size=(28, 28)):
    # Cargar imagen
    image = Image.open(image_path).convert("RGB")
    # Redimensionar imagen
    image_resized = image.resize(target_size)
    # Convertir a array de numpy
    image_array = np.array(image_resized)

    return image_array

def process_cropped_images(cropped_patches_df, cropped_images_dir):
    """ Procesa las imágenes cropped basándose en los metadatos proporcionados. """
    patches_data = []
    for index, row in cropped_patches_df.iterrows():
        # Obtener el codigo
        image_folder = row['CODI'] + "_1"
        image_density = row['DENSITAT']

        # Construye la ruta de la carpeta
        image_folder_path = os.path.join(cropped_images_dir, image_folder)
        
        if os.path.exists(image_folder_path):
            for filename in os.listdir(image_folder_path):
                image_path = os.path.join(image_folder_path, filename)
                image_array = load_and_process_image(image_path)

                if image_density == "NEGATIVA":
                    patches_data.append((image_array, -1))
                else:
                    patches_data.append((image_array, 1))
        else:
            #print(f"No se encontró la imagen: {image_folder_path}")
            u = 0

    return patches_data


def process_annotated_images(annotated_patches_df, annotated_images_dir, labeled_patients_df):
    #Procesa las imágenes anotadas basándose en los metadatos proporcionados.
    pacients_data = {}
    pacients_labels = {}
    for index, row in annotated_patches_df.iterrows():
        # Divide el patch_id en las partes necesarias
        patch_id_parts = row['ID'].split('.')
        image_folder = patch_id_parts[0]
        image_file = patch_id_parts[1] + '.png'
        patient_id = patch_id_parts[0][:-2]

        # Construye la ruta de la carpeta y el archivo de la imagen
        image_folder_path = os.path.join(annotated_images_dir, image_folder)
        image_path = os.path.join(image_folder_path, image_file)

        # Verifica si la ruta del archivo existe antes de procesar
        if os.path.exists(image_path):
            image_array = load_and_process_image(image_path)
            if patient_id not in list(pacients_data.keys()):
                pacients_data[patient_id] = []
                label_str = labeled_patients_df[labeled_patients_df["CODI"] == patient_id]["DENSITAT"].iloc[0]
                if label_str == "NEGATIVA":
                    label = -1
                else:
                    label = 1
                pacients_labels[patient_id] = label

            pacients_data[patient_id].append(image_array)

        else:
            print(f"No se encontró la imagen: {image_path}")
            
    return pacients_data, pacients_labels


def prepare_dataset(patches_data):
    """ Prepara el conjunto de datos para el entrenamiento. """
    patches_data_array = np.array([i[0] for i in patches_data])
    patches_labels_array = np.array([i[1] for i in patches_data])
    return patches_data_array, patches_labels_array


def normalize_tensors(data_tensor):
    """ Normalizar los tensores a un rango de [0, 1]. """
    return data_tensor / 255.0


def convert_to_tensors(patches_data_array, patches_labels_array):
    """ Convertir arrays de Numpy a tensores de PyTorch. """
    patches_data_array = patches_data_array.transpose((0, 3, 1, 2))
    data_tensor = torch.tensor(patches_data_array, dtype=torch.float32)
    labels_tensor = torch.tensor(patches_labels_array, dtype=torch.float32)
    return data_tensor, labels_tensor


def split_data(data_tensor, labels_tensor, test_size=0.2, random_state=42):
    """ Dividir los datos en conjuntos de entrenamiento y validación. """
    X_train, X_val, y_train, y_val = train_test_split(
        data_tensor, labels_tensor, test_size=test_size, random_state=random_state
    )
    return X_train, X_val, y_train, y_val


def create_dataloaders(X_train, y_train, X_val, y_val, batch_size=64):
    """ Crear DataLoaders para los conjuntos de entrenamiento y validación. """
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

#MAIN
def prep_data_main(annotated_patches_path, labeled_patients_path, annotated_images_dir, cropped_images_dir):

    # TRAINING AND VALIDATION DATALOADERS (SOLAMENTE IMAGENES NO INFECTADAS DEL DATASET CROPPED)
    # Cargar metadata
    labeled_patients_df = load_metadata(labeled_patients_path)
    
    # Separar los datos en infectados y no infectados
    non_infected_patches_df = labeled_patients_df[labeled_patients_df['DENSITAT'] == 'NEGATIVA']

    # Procesar imágenes cropped de pacientes no infectados
    non_infected_patches_data = process_cropped_images(non_infected_patches_df, cropped_images_dir)

    # Preparar el conjunto de datos de pacientes no infectados del dataset cropped
    non_infected_data_array,  non_infected_patches_labels_array = prepare_dataset(non_infected_patches_data)


    # DATASET PREPROCESSING

    # Convertir arrays de no infectados de Numpy a tensores de PyTorch
    non_infected_patches_data_tensor, non_infected_patches_labels_tensor = convert_to_tensors(non_infected_data_array, non_infected_patches_labels_array)

    # Normalizar los datos de no infectados si aún no lo están
    non_infected_patches_data_tensor = normalize_tensors(non_infected_patches_data_tensor)

    # Dividir los datos en entrenamiento y validación
    X_train, X_val, y_train, y_val = split_data(non_infected_patches_data_tensor, non_infected_patches_labels_tensor)

    # Crear DataLoaders
    non_infected_train_loader, non_infected_val_loader = create_dataloaders(X_train, y_train, X_val, y_val)
    

    # AQUI CREAREMOS Y GUARDAREMOS LOS DATOS QUE USAREMOS PARA VER SI NUESTRO METODO NOS CLASIFICA BIEN (HABRÁN TODAS LAS IMAGENES INFECTADAS / NO INFECTADAS DEL DATASET ANNOTATED)

    #Cargar metadata
    annotated_patches_df = load_metadata(annotated_patches_path)

    # Procesar imagenes anotadas de pacientes infectados / no infectados
    pacients_data, pacients_labels = process_annotated_images(annotated_patches_df, annotated_images_dir, labeled_patients_df)

    # To tensors
    pacients_data_tensors = {}
    for key, array_list in pacients_data.items():
        # Convertir la lista de arrays en un solo array numpy
        combined_array = np.array(array_list)
        # Transponer el array combinado
        transposed_array = combined_array.transpose((0, 3, 1, 2))
        # Convertir el array transpuesto en un tensor
        tensor = torch.tensor(transposed_array, dtype=torch.float32)
        # Guardar el tensor en el diccionario
        pacients_data_tensors[key] = tensor

    # Normalizar los datos de no infectados si aún no lo están
    pacients_data_tensors = {key: [tensor.float() / 255 for tensor in tensors] for key, tensors in pacients_data_tensors.items()}


    return non_infected_train_loader, non_infected_val_loader, pacients_data_tensors, pacients_labels




