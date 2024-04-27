import os
import torch
import json
from PIL import Image
from torch.utils.data import Dataset
from diffusers.models import AutoencoderKL
from streaming import MDSWriter

import logging
import time
import numpy as np
from typing import Any
from datasets import load_dataset

# Initialize logging
logging.basicConfig(level=logging.INFO)

from imagenet_labels import IMGNET_LABELS
from streaming.base.format.mds.encodings import Encoding, _encodings


class uint8(Encoding):
    def encode(self, obj: Any) -> bytes:
        return obj.tobytes()

    def decode(self, data: bytes) -> Any:
        return np.frombuffer(data, np.uint8)


class np16(Encoding):
    def encode(self, obj: Any) -> bytes:
        return obj.tobytes()

    def decode(self, data: bytes) -> Any:
        return np.frombuffer(data, np.float16)


_encodings["np16"] = np16
_encodings["uint8"] = uint8


def crop_to_center(image, new_size=768):
    width, height = image.size

    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))


def prepare_image(pil_image, w=512, h=512):
    pil_image = pil_image.resize((w, h), resample=Image.BICUBIC, reducing_gap=1)
    arr = np.array(pil_image.convert("RGB"))
    arr = arr.astype(np.float32) / 127.5 - 1
    arr = np.transpose(arr, [2, 0, 1])
    image = torch.from_numpy(arr)
    return image


class ImageDataset(Dataset):
    def __init__(self, idx_range=(0, 1000), is_test=False):

        self.dataset = load_dataset("imagenet-1k", split="train")
        self.idx_range = idx_range
        # modify the idx range if its out of bounds
        if self.idx_range[1] > len(self.dataset):
            self.idx_range = (self.idx_range[0], len(self.dataset))

    def __len__(self):
        return self.idx_range[1] - self.idx_range[0]

    def __getitem__(self, idx):
        idx = idx + self.idx_range[0]
        image, label = self.dataset[idx]["image"], self.dataset[idx]["label"]

        # image is pil. first center-crop to 256x256
        # for that, first resize to 256 x N or N x 256, then crop to 256 x 256

        w, h = image.size
        if w > h:
            image = image.resize((int(w * 256 / h), 256), resample=Image.BICUBIC)
        else:
            image = image.resize((256, int(h * 256 / w)), resample=Image.BICUBIC)

        image = crop_to_center(image, 256)
        image = prepare_image(image, 256, 256)

        return image, label


from tqdm import tqdm
from torch.utils.data import DataLoader


@torch.no_grad()
def convert_to_mds(
    idx_range, out_root, device, batch_size=8, num_workers=4, is_test=False
):
    logging.info(f"Processing on {device}")

    # Load the VAE model
    vae_model = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
    vae_model = vae_model.to(device).eval()
    vae_model.to(memory_format=torch.channels_last)
    # vae_model.encode = torch.compile(vae_model.encode, mode="reduce-overhead", fullgraph=False)

    # Create the dataset and dataloader
    dataset = ImageDataset(idx_range, is_test=is_test)
    
    if dataset.__len__() < 1:
        logging.info("No images to process.")
        return

    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True
    )

    sub_data_root = os.path.join(out_root, "data")
    columns = {"vae_output": "uint8", "label": "str", "label_as_text" : 'str'}

    if os.path.exists(sub_data_root):
        # Remove all files in the directory
        for file in os.listdir(sub_data_root):
            os.remove(os.path.join(sub_data_root, file))
    os.makedirs(sub_data_root, exist_ok=True)

    with MDSWriter(out=sub_data_root, columns=columns) as out:
        inference_latencies = []

        for batch in tqdm(dataloader):
            start_time = time.time()

            processed_images, labels = batch
            processed_images = processed_images.to(device)
            vae_outputs = vae_model.encode(processed_images).latent_dist.sample()
            
            vae_outputs = (vae_outputs.clip(-14, 14) / 28.0 + 0.5) * 255.0
            vae_outputs = vae_outputs.to(torch.uint8)

            # Iterate through the batch
            for i in range(len(labels)):
                sample = {
                    "vae_output": vae_outputs[i].cpu().numpy().astype(np.uint8),
                    "label": str(labels[i].item()),
                    "label_as_text": IMGNET_LABELS[labels[i].item()],
                }
                out.write(sample)

            inference_latencies.append(time.time() - start_time)

        logging.info(
            f"Average Inference Latency on {device}: {np.mean(inference_latencies)} seconds"
        )


def main(
    root_dir, out_root, batch_size=64, num_workers=8, is_test=False, device_name="cuda"
):
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    print(f"Processing on {device}")
    convert_to_mds(root_dir, out_root, device, batch_size, num_workers, is_test=is_test)
    logging.info("Finished processing images.")


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert images to MDS format.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for processing (cuda or cpu).",
    )
    parser.add_argument(
        "--file_index", type=int, default=0, help="File index to process."
    )
    parser.add_argument(
        "--is_test", action="store_true", help="Run in test mode with reduced dataset."
    )

    args = parser.parse_args()

    out_root = f"./vae_mds/{str(args.file_index).zfill(5)}"

    main(
        (args.file_index * 20000, (args.file_index + 1) * 20000),
        out_root,
        is_test=args.is_test,
        device_name=args.device,
    )
