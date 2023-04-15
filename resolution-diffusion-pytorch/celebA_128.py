from resolution_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import torchvision
import argparse
from PIL import Image, ImageFilter

parser = argparse.ArgumentParser()
parser.add_argument('--time_steps', default=50, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--train_steps', default=700000, type=int)
parser.add_argument('--save_folder', default='./results_celebA', type=str)
parser.add_argument('--load_path', default=None, type=str)
parser.add_argument('--data_path', default='./root_celebA_128_train_new/', type=str)
parser.add_argument('--resolution_routine', default='Incremental', type=str)
parser.add_argument('--train_routine', default='Final', type=str)
parser.add_argument('--sampling_routine', default='x0_step_down', type=str)
parser.add_argument('--wave_type', default='haar', type=str)
parser.add_argument('--remove_time_embed', action="store_true")
parser.add_argument('--residual', action="store_true")
parser.add_argument('--shrink_waves', action="store_true")
parser.add_argument('--dbg', action="store_true")

args = parser.parse_args()
print(args)

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels=3,
    with_time_emb=not(args.remove_time_embed),
    residual=args.residual
).cuda()

import torch
from torchvision import transforms
from pathlib import Path
class Dataset(torch.utils.data.Dataset):
    def __init__(self, folder, image_size, exts = ['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.transform = transforms.Compose([
            transforms.Resize((int(image_size*1.12), int(image_size*1.12))),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

image_size = 128

ds = Dataset(folder=args.data_path, image_size=image_size)
diffusion = GaussianDiffusion(
    model,
    image_size = image_size,
    device_of_kernel = 'cuda',
    channels = 3,
    timesteps = args.time_steps,        # number of steps
    loss_type = 'l1',                   # L1 or L2
    resolution_routine=args.resolution_routine,
    train_routine = args.train_routine,
    sampling_routine = args.sampling_routine,
    dataset=ds,
    wave_type =args.wave_type,
    shrink_waves=args.shrink_waves
).cuda()

import torch
diffusion = torch.nn.DataParallel(diffusion, device_ids=range(torch.cuda.device_count()))

if args.dbg:
    shuffle=False
    num_workers=0
else:
    shuffle=True
    num_workers=8

trainer = Trainer(
    diffusion,
    args.data_path,
    image_size = image_size,
    train_batch_size = args.batch_size,
    train_lr = 2e-5,
    train_num_steps = args.train_steps, # total training steps
    save_and_sample_every = 1000,
    gradient_accumulate_every = 2,      # gradient accumulation steps
    ema_decay = 0.995,                  # exponential moving average decay
    fp16 = False,                       # turn on mixed precision training with apex
    results_folder = args.save_folder,
    load_path = args.load_path,
    dataset = 'celebA',
    shuffle=shuffle,
    num_workers=num_workers
)

trainer.train()
