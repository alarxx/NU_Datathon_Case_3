import torch
from torchvision import transforms
from torchvision.models import vgg16
import torchvision.models as models
from PIL import Image
import csv
import os

def preprocess_image(img_path):
    """Load and preprocess the image."""
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor

def make_prediction(model, img_tensor):
    """Make a prediction on a new image."""
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

if __name__ == "__main__":
    # Load the saved model
    # model = torch.load("model_fold5.pth")
    model = models.vgg16() # we do not specify ``weights``, i.e. create untrained model
    
    # model.classifier[6] = torch.nn.Linear(4096, 2)
    # model.load_state_dict(torch.load("model_fold5.pth"))

    # model.load_state_dict(torch.load("model_eff_fold2.pth"))
    # model = torch.load("model_eff_fold2.pth")
    # model = model.cpu()

    model.classifier[6] = torch.nn.Linear(4096, 5)
    model.load_state_dict(torch.load("multiclass-model.pth"))
    
    model.eval()

    images = ["23344641.jpeg", "23389951.jpeg", "23390645.jpeg", "23516653.jpeg", "почему ты один.jpeg"]
    for img_path in images:
        full_img_path = os.path.join("examples", img_path)
        img_tensor = preprocess_image(full_img_path)
        prediction = make_prediction(model, img_tensor)
        print(f"{img_path} {prediction}")