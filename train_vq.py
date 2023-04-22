from tqdm import tqdm
from torch import nn, optim
import torch, argparse, math
from model.vq_model import VQModel
from lossers.lpips import LPIPS
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from torchvision.transforms import functional as F
from model.discriminator import NLayerDiscriminator
from lossers.gan import (
    hinge_d_loss as d_loss_fn,
    vanilla_g_loss as g_loss_fn,
)
from utils import sample_data, requires_grad

parser = argparse.ArgumentParser(description="Train VQModel")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--dataset", type=str, default="lambdalabs/pokemon-blip-captions")
parser.add_argument("--cache_dir", type=str, default="./.cache")
parser.add_argument("--iter", type=int, default=800000, help="total training iterations")
parser.add_argument("--batch", type=int, default=64, help="batch sizes for each gpus")
parser.add_argument("--size", type=int, default=64, help="image sizes for the model")
args = parser.parse_args()

model = VQModel().to(args.device)
lpips = LPIPS(net='vgg', cache_dir=args.cache_dir).to(args.device)
discriminator = NLayerDiscriminator().to(args.device)
vq_optim = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0, 0.999))
d_optim = torch.optim.Adam(discriminator.parameters(), lr=1e-3, betas=(0, 0.999))

to_tensor = transforms.Compose([
    transforms.Resize(args.size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
])
def preprocess(data):
    for i in range(len(data['image'])):
        data['image'][i] = to_tensor(data['image'][i])
    return data
dataset = load_dataset(args.dataset, split="train", cache_dir=args.cache_dir)
dataset = dataset.with_transform(preprocess)
dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True, drop_last=True)
dataloader = sample_data(dataloader)
pbar = tqdm(range(args.iter))
sample = next(dataloader)['image']

for idx in pbar:
    image = next(dataloader)['image'].to(args.device)
    # train vqmodel
    requires_grad(model, True)
    requires_grad(discriminator, False)
    rec = model(image)
    lpips_loss = lpips(image, rec).mean()
    fake_pred = discriminator(rec)
    g_loss = g_loss_fn(fake_pred)
    loss = lpips_loss + g_loss
    vq_optim.zero_grad()
    loss.backward()
    vq_optim.step()
    # train discriminator
    requires_grad(model, False)
    requires_grad(discriminator, True)
    real_pred, fake_pred = discriminator(image), discriminator(rec.detach())
    d_loss = d_loss_fn(real_pred, fake_pred)
    d_optim.zero_grad()
    d_loss.backward()
    d_optim.step()
    pbar.set_description(
        f'lpips_loss: {lpips_loss.item():.4f}, g_loss: {g_loss.item():.4f}, '
        f'd_loss: {d_loss.item():.4f}'
    )

    if idx % 200 == 0:
        with torch.no_grad():
            rec = model(sample)
            utils.save_image(
                rec, f"sample/{str(idx).zfill(6)}.png",
                nrow=int(math.sqrt(args.batch)), normalize=True, value_range=(-1, 1),
            )
