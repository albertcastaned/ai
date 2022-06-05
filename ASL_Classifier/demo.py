import torch
import torchvision

import cv2
import numpy as np
import torchvision.transforms as transforms

from models.resnet import ResNet

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

model = ResNet(3, 29)
if torch.cuda.is_available():
    model.load_state_dict(torch.load('resnet.pth'))
else:
    model.load_state_dict(torch.load('resnet.pth', map_location='cpu'))

model.eval()

capture = cv2.VideoCapture(0)

capture.set(3, 700)
capture.set(4, 480)


while True:
    ret, frame = capture.read()

    offset_x = 224
    offset_y = 224

    start_x = offset_x + 0
    end_x = 224 + offset_x

    start_y = offset_y + 0
    end_y = offset_y + 224

    img = frame[start_x:end_x, start_y: end_y]
    img = cv2.resize(img, (32, 32))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    input_data = transform(img)

    with torch.no_grad():
        out = model(input_data.view(1, 3, 32, 32))
        _, top_predictions = torch.topk(out, 3)
        prediction_label = ""
        for index, prediction in enumerate(top_predictions[0]):\
            prediction_label += f"{index + 1}. {classes[int(prediction)]}   "

    # Font and text settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2

    image = cv2.putText(frame, prediction_label, org, font, 
                    fontScale, color, thickness, cv2.LINE_AA)

    frame = cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Sign Language Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()