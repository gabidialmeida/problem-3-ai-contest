import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# ---- Model ----
# THIS GENERATOR CLASS MUST EXACTLY MATCH THE ONE USED FOR TRAINING
class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        x = torch.cat((noise, self.label_emb(labels)), -1)
        out = self.model(x)
        return out.view(-1, 1, 28, 28)

# ---- Load model ----
# It's good practice to keep the device explicitly 'cpu' for Streamlit deployments
# unless you specifically have a GPU set up on your deployment environment.
device = torch.device('cpu')
latent_dim = 100
num_classes = 10

# Initialize the Generator with the correct architecture
G = Generator(latent_dim, num_classes).to(device)

# Load the trained state_dict
try:
    G.load_state_dict(torch.load('generator_mnist_improved.pth', map_location=device))
    G.eval() # Set model to evaluation mode (important for BatchNorm and Dropout layers)
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error("Error: 'generator_mnist_improved.pth' not found. Please ensure the model file is in the same directory.")
    st.stop() # Stop the app if the model isn't found
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ---- Function to generate images ----
def generate_digit_images(digit, n_images=5):
    noise = torch.randn(n_images, latent_dim).to(device)
    labels = torch.full((n_images,), digit, dtype=torch.long).to(device)
    with torch.no_grad():
        imgs = G(noise, labels).cpu()
    # GAN outputs are usually Tanh-activated, meaning values are in [-1, 1].
    # For PIL/matplotlib, we need them in [0, 1].
    imgs = (imgs + 1) / 2
    return imgs

# ---- Streamlit App ----
st.title("Digit Generator - MNIST GAN")
st.write("Generate synthetic handwritten digits using a trained Conditional GAN model.")

digit = st.selectbox("Choose a digit to generate (0-9):", list(range(10)))

if st.button("Generate Images"):
    st.write(f"Generating 5 images of the digit: **{digit}**")
    
    # Add a spinner while generating
    with st.spinner('Generating images...'):
        imgs = generate_digit_images(digit)

    # Display images side by side
    cols = st.columns(len(imgs)) # Create columns for each image
    for i, col in enumerate(cols):
        # Convert tensor to numpy array, squeeze the channel dimension (1, 28, 28) -> (28, 28)
        img_np = imgs[i].squeeze().numpy()
        
        # Convert to PIL Image for display.
        # np.uint8 is crucial as image display functions expect 0-255 pixel values.
        # 'L' mode means 8-bit pixels, black and white.
        img_pil = Image.fromarray(np.uint8(img_np * 255), 'L')
        
        col.image(img_pil, caption=f"Generated {digit} - Image {i+1}", width=150)

st.markdown("---")
st.markdown("This application uses a Conditional Generative Adversarial Network (CGAN) trained on the MNIST dataset.")