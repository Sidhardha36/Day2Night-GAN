import torch
from PIL import Image
from torchvision import transforms
from models.generator import GeneratorUNet
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------
# Load Generator
# ------------------------
generator = GeneratorUNet().to(device)

# Load last checkpoint (G_epoch_10.pth)
checkpoint = "outputs/G_epoch_10.pth"
generator.load_state_dict(torch.load(checkpoint, map_location=device))
generator.eval()

print("Loaded model:", checkpoint)

# ------------------------
# Preprocessing (match training)
# ------------------------
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5)),
])

# ------------------------
# Predict Function
# ------------------------
def generate_night(input_path):
    img = Image.open(input_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        fake = generator(img)[0]

    # Denormalize
    fake = (fake * 0.5) + 0.5
    fake = torch.clamp(fake, 0, 1)

    os.makedirs("outputs/inference", exist_ok=True)
    output_path = "outputs/inference/" + os.path.basename(input_path)

    transforms.ToPILImage()(fake.cpu()).save(output_path)
    print("Saved:", output_path)


# ------------------------
# Run demo
# ------------------------
if __name__ == "__main__":
    test_image = "test_A/0a42ee8c-13838721.jpg"   # put your day image here
    generate_night(test_image)
