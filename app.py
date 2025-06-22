import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# ---- Model ----
class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        x = torch.cat((noise, self.label_emb(labels)), -1)
        return self.model(x).view(-1, 1, 28, 28)

# ---- Load model ----
device = torch.device('cpu')
latent_dim = 100
num_classes = 10
G = Generator(latent_dim, num_classes).to(device)
G.load_state_dict(torch.load('generator_mnist_improved.pth', map_location=device))
G.eval()

# ---- Function to generate images ----
def generate_digit_images(digit, n_images=5):
    noise = torch.randn(n_images, latent_dim).to(device)
    labels = torch.full((n_images,), digit, dtype=torch.long).to(device)
    with torch.no_grad():
        imgs = G(noise, labels).cpu()
    imgs = (imgs + 1) / 2  # scale to [0,1]
    return imgs

# ---- Streamlit App ----
st.title("Digit Generator - MNIST GAN")

digit = st.selectbox("Choose a digit to generate (0-9):", list(range(10)))

if st.button("Generate Images"):
    st.write(f"Generating 5 images of the digit: {digit}")
    imgs = generate_digit_images(digit)

    # Display images side by side
    cols = st.columns(len(imgs))
    for i, col in enumerate(cols):
        img_np = imgs[i].squeeze().numpy()
        img_pil = Image.fromarray(np.uint8(img_np * 255), 'L')
        col.image(img_pil, caption=f"Digit {digit} - Image {i+1}", width=150)
