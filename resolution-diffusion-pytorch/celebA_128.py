import logging

import torch
from torchvision import transforms
from pathlib import Path
from resolution_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import torchvision
import argparse
import os
from PIL import Image, ImageFilter

logging.basicConfig(level=logging.INFO)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, folder, image_size, exts=['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder = folder
        assert os.path.isdir(self.folder)
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{self.folder}').glob(f'**/*.{ext}')]

        self.transform = transforms.Compose([
            transforms.Resize((int(image_size * 1.12), int(image_size * 1.12))),
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

import datetime
import clearml
def run_trainer_celeba128():
    log = logging.getLogger('TRAINER-Celeb18')
    log.info('Run ResDiff Trainer for Celeba128')

    experiment_base_name = 'train-res-diff-celeba'
    parser = argparse.ArgumentParser('ResDiff parameters')
    parser.add_argument('--experiment', default='', type=str)
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
    parser.add_argument('--unet_dim', default=64, type=int)
    parser.add_argument('--channels', default=3, type=int)
    parser.add_argument('--unet_deep', default=4, type=int)
    parser.add_argument('--unet_layer_multi', default=2, type=int)
    parser.add_argument('--loss_type', default='l1', type=str)
    parser.add_argument('--image_size', default=128, type=int)
    parser.add_argument('--train_lr', default=2e-5, type=float)

    parser.add_argument('--save_and_sample_every', default=100, type=int)
    parser.add_argument('--gradient_accumulate_every', default=2, type=int)
    parser.add_argument('--ema_decay', default=0.995, type=float),  # gradient accumulation steps
    parser.add_argument('--dataset', default='celebA', type=str)
    parser.add_argument('--num_workers', default=8, type=int)

    args = parser.parse_args()
    log.info(args)

    experiment_name = experiment_base_name + args.experiment
    time_stamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    task_name= f'{experiment_name}-{time_stamp}'
    task = clearml.Task.init(task_name=task_name, project_name='NAD')
    clearml.Logger.current_logger().set_default_upload_destination('s3://data-clearml')
    task.upload_artifact('config-trainer', args.__dict__)

    dim_mults = [ 2**args.unet_layer_multi for i in range(args.unet_deep)]
    model = Unet(
        dim=args.unet_dim,
        dim_mults=dim_mults,
        channels=args.channels,
        with_time_emb=not (args.remove_time_embed),
        residual=args.residual
    ).cuda()


    ds = Dataset(folder=args.data_path, image_size=args.image_size)
    diffusion = GaussianDiffusion(
        model,
        image_size=args.image_size,
        device_of_kernel='cuda',
        channels=args.channels,
        timesteps=args.time_steps,  # number of steps
        loss_type=args.loss_type,  # L1 or L2
        resolution_routine=args.resolution_routine,
        train_routine=args.train_routine,
        sampling_routine=args.sampling_routine,
        dataset=ds,
        wave_type=args.wave_type,
        shrink_waves=args.shrink_waves
    ).cuda()

    diffusion = torch.nn.DataParallel(diffusion, device_ids=range(torch.cuda.device_count()))

    if args.dbg:
        shuffle = False
        num_workers = 0
    else:
        shuffle = True
        num_workers = args.num_workers

    trainer = Trainer(
        diffusion,
        args.data_path,
        image_size=args.image_size,
        train_batch_size=args.batch_size,
        train_lr=args.train_lr,
        train_num_steps=args.train_steps,  # total training steps
        save_and_sample_every=args.save_and_sample_every,
        gradient_accumulate_every=args.gradient_accumulate_every,  # gradient accumulation steps
        ema_decay=args.ema_decay,  # exponential moving average decay
        fp16=False,  # turn on mixed precision training with apex
        results_folder=args.save_folder,
        load_path=args.load_path,
        dataset=args.dataset,
        shuffle=shuffle,
        num_workers=num_workers
    )

    trainer.train()

if __name__ == '__main__':
    run_trainer_celeba128()