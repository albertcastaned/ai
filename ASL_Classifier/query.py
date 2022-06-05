from PIL import Image
import torch
import numpy as np
from models.resnet import ResNet
import torchvision.transforms as transforms

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

model = ResNet(3, 29)
if torch.cuda.is_available():
    model.load_state_dict(torch.load('resnet.pth'))
else:
    model.load_state_dict(torch.load('resnet.pth', map_location='cpu'))
model.eval()


while True:
    path = input("Image path (relative):    ")
    img = Image.open(path).convert('RGB')
    img = img.resize((32, 32))

    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    input_data = transform(img)
    input_data = input_data.type(torch.float32)


    with torch.no_grad():
        out = model(input_data.view(1, 3, 32, 32))
        _, prediction = torch.max(out, dim=1)
        prediction_label = classes[int(prediction)]

    print(f"Prediction: {prediction_label}")

    choice = input("Do another prediction? (Y \ N):     ")

    if str(choice) == 'Y':
        continue
    else:
        break