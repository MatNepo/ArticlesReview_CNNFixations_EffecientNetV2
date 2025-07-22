import json

# Загрузите метки классов
with open('imagenet-simple-labels.json', 'r') as f:
    labels = json.load(f)

# Печать предсказанного класса с меткой
predicted_class, confidence = predict(image_path)
print(f'Предсказанный класс: {labels[predicted_class]}, Уверенность: {confidence}')
