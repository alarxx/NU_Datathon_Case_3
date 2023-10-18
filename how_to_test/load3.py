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
        transforms.Resize((480, 480)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor

def make_prediction(model, img_tensor):
    """Make a prediction on a new image."""
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

if __name__ == "__main__":
    # Load the saved model
    model = torch.load("effy3.pth")
    model.eval()
    model = model.cpu()

    with open("effy_3.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['file_index', 'class'])  # writing header
        test_images_dir = "test"
        # Directory containing your test images
        for img_name in os.listdir(test_images_dir):
            print(img_name)
            img_path = os.path.join(test_images_dir, img_name)
            img_tensor = preprocess_image(img_path)
            prediction = make_prediction(model, img_tensor)
            writer.writerow([os. path. splitext(img_name)[0], prediction])