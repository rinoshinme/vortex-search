import numpy as np
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from PIL import Image


class Resnet50Extractor(object):
    def __init__(self):
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.model = torchvision.models.resnet50(pretrained=True)
        self.model.fc = nn.Identity()
        self.model.eval()
        self.model.to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                                 std=(0.229, 0.224, 0.225))
        ])
    
    def infer(self, image_path):
        image = Image.open(image_path)
        image = image.convert('RGB')
        inputs = self.transform(image)
        inputs = inputs.unsqueeze(0)
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(inputs)
        outputs = outputs[0].detach().cpu().numpy()
        index = np.argmax(outputs)
        return index
    
    def extract(self, image_path, normalize=True):
        image = Image.open(image_path)
        image = image.convert('RGB')
        inputs = self.transform(image)
        inputs = inputs.unsqueeze(0)
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(inputs)
        outputs = outputs.detach().cpu().numpy()
        outputs = outputs[0]
        if normalize:
            length = np.linalg.norm(outputs)
            outputs = outputs / length
        return outputs
    

if __name__ == '__main__':
    image_path = '../../data/cat3.jpg'
    model = Resnet50()
    outputs = model.extract(image_path)
    print(outputs.shape)
