from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import torch
import time

# Load the model (on GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
).to(device)

# Load your image and resize to 512x512
init_image = Image.open("input.png").convert("RGB").resize((512, 512))

# Describe how you want the output to look
prompt = "turn it into a zebra"

# Start timer
start_time = time.time()
# Generate image
result = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images[0]
# End timer
end_time = time.time()

# Save result
result.save("output.png")

print("Image saved as output.png")


print(f"Inference time: {end_time - start_time:.2f} seconds")
