
import os
import torch
import json
from PIL import Image
import logging
import time
import numpy as np
from diffusers.models import AutoencoderKL
from diffusers import StableDiffusionPipeline
from diffusers.image_processor import VaeImageProcessor



from diffusers.models import AutoencoderKL

def prepare_image(pil_image, w=256, h=512):
    pil_image = pil_image.resize((w, h), resample=Image.BICUBIC, reducing_gap=1)
    arr = np.array(pil_image.convert("RGB"))
    arr = arr.astype(np.float32) / 127.5 - 1
    arr = np.transpose(arr, [2, 0, 1])
    image = torch.from_numpy(arr)
    return image


device = "cuda:0"
vae_model = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
vae_model = vae_model.to(device).eval()
vae_model.to(memory_format=torch.channels_last)



img = Image.open("tiger.jpg")
w, h  = img.size
if w > h:
    img = img.crop((w//2 - h//2, 0, w//2 + h//2, h))
else:
    img = img.crop((0, h//2 - w//2, w, h//2 + w//2))

imgorg = img.resize((256, 256), resample=Image.BICUBIC)
imgorg.save("tiger.png")

save_name = "tiger"

for precision in [torch.float32, torch.float16, torch.uint8]:

    img = imgorg.resize((256, 256), resample=Image.BICUBIC)
    img = prepare_image(img, 256, 256)
    img = img.to(device)

    latent = vae_model.encode(img.unsqueeze(0)).latent_dist.sample()

    # based on the precision do action
    if precision in [torch.float32, torch.float16]:
        latent = latent.to(precision)
        # back to full
        latent = latent.to("cpu").to(torch.float32)
        print("fp32 or fp16")
    elif precision == torch.uint8:
        latent = (latent.clip(-14, 14) / 28.0 + 0.5) * 255.0
        latent = latent.to(torch.uint8)
        # back to full
        latent = latent.to("cpu").to(torch.float32)
        latent = (latent / 255.0 - 0.5) * 28.0
        print("uint8")

    # decode
    latent = latent.to(device)
    x = vae_model.decode(latent).sample
    img = VaeImageProcessor().postprocess(image = x.detach(), do_denormalize = [True, True])[0]

    img.save(f"{save_name}_{precision}.png")
