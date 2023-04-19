from tqdm import tqdm
from torch import nn, optim
import torch, argparse, math
from vq_model import VQModel
from lossers.lpips import LPIPS
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from torchvision.transforms import functional as F

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

parser = argparse.ArgumentParser(description="Train VQModel")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--dataset", type=str, default="lambdalabs/pokemon-blip-captions")
parser.add_argument("--cache_dir", type=str, default="./.cache")
parser.add_argument("--iter", type=int, default=800000, help="total training iterations")
parser.add_argument("--batch", type=int, default=64, help="batch sizes for each gpus")
parser.add_argument("--size", type=int, default=64, help="image sizes for the model")
args = parser.parse_args()

model = VQModel().to(args.device)
lpips = LPIPS(net='vgg').to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))

to_tensor = transforms.Compose([
    transforms.Resize(args.size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
])
def preprocess(data):
    data['image'][0] = to_tensor(data['image'][0]).to(args.device)
    return data
dataset = load_dataset(args.dataset, split="train", cache_dir=args.cache_dir)
dataset = dataset.with_transform(preprocess)
dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True, drop_last=True)
dataloader = sample_data(dataloader)
pbar = tqdm(range(args.iter))
sample = next(dataloader)['image']

for idx in pbar:
    image = next(dataloader)['image']
    rec = model(image)
    loss = lpips(image, rec).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    pbar.set_description((f"loss: {loss.item():.4f}"))

    if idx % 200 == 0:
        with torch.no_grad():
            rec = model(sample)
            utils.save_image(
                rec, f"sample/{str(idx).zfill(6)}.png",
                nrow=int(math.sqrt(args.batch)), normalize=True, value_range=(-1, 1),
            )
