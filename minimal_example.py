
# do huggingface-cli download --repo-type dataset cloneofsimo/imagenet.int8 --local-dir ./vae_mds

from streaming.base.format.mds.encodings import Encoding, _encodings
import numpy as np
from typing import Any
import torch
from streaming import StreamingDataset
from diffusers.models import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor

class uint8(Encoding):
    def encode(self, obj: Any) -> bytes:
        return obj.tobytes()

    def decode(self, data: bytes) -> Any:
        x=  np.frombuffer(data, np.uint8).astype(np.float32)
        return (x / 255.0 - 0.5) * 24.0

_encodings["uint8"] = uint8


remote_train_dir = "./vae_mds"
local_train_dir = "./local_train_dir"

train_dataset = StreamingDataset(
    local=local_train_dir,
    remote=remote_train_dir,
    split=None,
    shuffle=True,
    shuffle_algo="naive",
    num_canonical_nodes=1,
    batch_size = 32
)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    num_workers=3,
)


model = "stabilityai/your-stable-diffusion-model"
vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to("cuda:0")


batch = next(iter(train_dataloader))

i = 5
vae_latent = batch["vae_output"].reshape(-1, 4, 32, 32)[i:i+1].cuda().float()
idx = batch["label"][i]
text_label = batch['label_as_text'][i]

print(f"idx: {idx}, text_label: {text_label}, latent: {vae_latent.shape}")
# idx: 402, text_label: acoustic guitar, latent: torch.Size([1, 4, 32, 32])

# example decoding
x = vae.decode(vae_latent.cuda()).sample
img = VaeImageProcessor().postprocess(image = x.detach(), do_denormalize = [True, True])[0]
img.save("5th_image.png")


