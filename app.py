import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from models.generator import GeneratorUNet
import os


# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Generator
generator = GeneratorUNet().to(device)
checkpoint = "outputs/G_epoch_10.pth"
generator.load_state_dict(torch.load(checkpoint, map_location=device))
generator.eval()

# Preprocessing (same as training)
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5)),
])

def generate_night(img):
    """Convert daytime image → night using generator"""
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        fake = generator(img_tensor)[0]

    # Denormalize
    fake = (fake * 0.5) + 0.5
    fake = torch.clamp(fake, 0, 1)

    return transforms.ToPILImage()(fake.cpu())



# Streamlit UI
st.title("🌙 Day → Night Image Translation (Pix2Pix GAN)")
st.markdown("Upload a daytime image and convert it into night-time using your trained GAN model.")

uploaded_file = st.file_uploader("Upload a Day Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    input_image = Image.open(uploaded_file).convert("RGB")

    st.subheader("📸 Input Image")
    st.image(input_image, width=300)

    if st.button("Convert to Night 🌙"):
        with st.spinner("Generating night image... please wait"):
            output_image = generate_night(input_image)

        st.subheader("🌌 Output Night Image")
        st.image(output_image, width=300)

        # Save output
        output_path = "outputs/inference/streamlit_output.jpg"
        os.makedirs("outputs/inference", exist_ok=True)
        output_image.save(output_path)

        st.success("Image generated successfully!")

        # Download button
        with open(output_path, "rb") as f:
            btn = st.download_button(
                label="Download Night Image",
                data=f,
                file_name="night_image.jpg",
                mime="image/jpeg"
            )
