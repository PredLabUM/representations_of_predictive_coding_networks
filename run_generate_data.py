import os
import os.path as op

import matplotlib.pyplot as plt
import torch
from diffusers import DiffusionPipeline


# Define image generation settings.
n_images = 5
n_iterations = 20
height, width = (None, None)
num_inference_steps = 40
save_id = "ext4"
save_images = True
plot_images = False
print(f"Total added images per class: {n_images * n_iterations}")

# Set directories.
base_dir = "/scratch/guetlid95/datasets/mcdermott_2024/Stimuli_extended"
subfolders = [d for d in os.listdir(base_dir) if op.isdir(op.join(base_dir, d))]

# Load the diffusion pipeline.
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-base")
pipe = pipe.to("cuda")

# Generate images for all classes.
for subfolder in subfolders:
    obj = subfolder.replace("_", " ")
    obj = "church interior" if obj == "church" else obj  # Avoid outside photos of churches.
    prompt = f"A typical photograph of a {obj} taken with a modern DSLR camera."
    print(prompt)
    negative_prompt = "Unrealistic, cartoon, art, anatomically incorrect, biologically incorrect, unreal, fantasy, black and white, exaggerated proportions, unrealistic lighting, over-saturated colors, cinematic effects, fantasy elements, surreal features, dream-like qualities, fantasy architecture, artificial looking textures, over-stylized compositions, dramatic lighting"

    # Generate images for the current class.
    img_count = 0
    for i in range(n_iterations):
        images = pipe(prompt, num_images_per_prompt=n_images, height=height, width=width,
                      num_inference_steps=num_inference_steps, negative_prompt=negative_prompt).images

        for j, img in enumerate(images):
            img_count += 1
            if save_images:
                save_path = op.join(base_dir, subfolder, f"{subfolder}_{save_id}_{img_count}.png")
                img.save(save_path, format="PNG")
                print("saved under: ", save_path)

            if plot_images:
                plt.imshow(img)
                plt.axis("off")
                plt.show()