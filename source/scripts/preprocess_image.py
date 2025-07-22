from PIL import Image
import torch
from torchvision import transforms


# Функция для предобработки изображения
def preprocess_image(image_path):
    # Открываем изображение
    image = Image.open(image_path).convert('RGB')

    # Применяем те же преобразования, что использовались при обучении
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Преобразуем изображение
    image_tensor = preprocess(image).unsqueeze(0)  # Добавляем batch dimension
    return image_tensor.cuda()
