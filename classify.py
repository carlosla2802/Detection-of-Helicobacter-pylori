
import torch 
from sklearn.metrics import classification_report
from skimage.color import rgb2hsv

def evaluate_reconstructions(model, images_list, device):
    model.eval()
    reconstructions = []
    with torch.no_grad():
        for image in images_list:
            image = image.to(device)
            reconstructed = model(image.unsqueeze(0))  # Añade una dimensión de lote (batch) de 1
            reconstructions.append(reconstructed.squeeze().cpu())
    return reconstructions

def detect_h_pylori(original_images, reconstructed_images):

    # Calcular la fracción de píxeles con tonalidades de color similares al rojo en las imágenes originales y reconstruidas
    red_fraction_original = compute_red_fraction(original_images)
    red_fraction_reconstructed = compute_red_fraction(reconstructed_images)

    # Calcular la diferencia en la fracción de píxeles con tonalidades de color similares al rojo
    red_fraction_difference = red_fraction_original - red_fraction_reconstructed

    # Etiquetar la ventana como que contiene H. pylori si la fracción perdida (red_fraction_difference) es mayor que 1
    if red_fraction_difference > 1:
        return 1
    else:
        return -1
    

# Función para calcular la fracción de píxeles con tonalidades de color similares al rojo en una imagen
def compute_red_fraction(image):
    # Reinvertir normalizacion
    image = image * 255

    # Convertir la imagen de RGB a HSV
    hsv_image = rgb2hsv(image.permute(1, 2, 0).numpy())

    # Filtrar píxeles con tonalidades de color similares al rojo (en el rango [-20, 20] en el espacio de tonalidades)
    red_like_pixels_filter = (hsv_image[..., 0] >= -20 / 360) & (hsv_image[..., 0] <= 20 / 360)

    # Calcular la fracción de píxeles con tonalidades de color similares al rojo
    red_like_pixels = red_like_pixels_filter.sum()

    red_fraction = red_like_pixels / len(image)

    return red_fraction


def detect_h_pylori_all_validation(validation_patches_data_tensor, reconstructed_validation_patches_data_tensor):
    pred_labels = []
    for original_image, reconstructed_image in zip(validation_patches_data_tensor, reconstructed_validation_patches_data_tensor):
        label = detect_h_pylori(original_image, reconstructed_image)
        pred_labels.append(label)

    return pred_labels


def calculate_metrics(real_labels, pred_labels):
    report = classification_report(real_labels, pred_labels)
    return report