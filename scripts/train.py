from mel2wav.dataset import AudioDataset
from mel2wav.modules import Generator, Discriminator, Audio2Mel
from mel2wav.utils import save_sample

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import yaml
import numpy as np
import time
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", default="logs/baseline")
    parser.add_argument("--load_path", default=None)

    parser.add_argument("--n_mel_channels", type=int, default=80)
    parser.add_argument("--ngf", type=int, default=32)
    parser.add_argument("--n_residual_layers", type=int, default=3)

    parser.add_argument("--ndf", type=int, default=16)
    parser.add_argument("--num_D", type=int, default=3)
    parser.add_argument("--n_layers_D", type=int, default=4)
    parser.add_argument("--downsamp_factor", type=int, default=4)
    parser.add_argument("--lambda_feat", type=float, default=10)
    parser.add_argument("--cond_disc", action="store_true")

    parser.add_argument("--data_path", default="/Users/zhangsan/Workspace/vocoder_class/melgan-neurips", type=Path)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=8192)

    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--n_test_samples", type=int, default=8)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    root = Path(args.save_path)
    load_root = Path(args.load_path) if args.load_path else None
    root.mkdir(parents=True, exist_ok=True)

    ####################################
    # Dump arguments and create logger #
    ####################################
    with open(root / "args.yml", "w") as f:
        yaml.dump(args, f)
    writer = SummaryWriter(str(root))

    #######################
    # Load PyTorch Models #
    #######################
    netG = Generator(args.n_mel_channels, args.ngf, args.n_residual_layers)
    netD = Discriminator(
        args.num_D, args.ndf, args.n_layers_D, args.downsamp_factor
    )
    fft = Audio2Mel(n_mel_channels=args.n_mel_channels)

    # print(netG)
    # print(netD)

    #####################
    # Create optimizers #
    #####################
    optG = torch.optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optD = torch.optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))

    if load_root and load_root.exists():
        netG.load_state_dict(torch.load(load_root / "netG.pt"))
        optG.load_state_dict(torch.load(load_root / "optG.pt"))
        netD.load_state_dict(torch.load(load_root / "netD.pt"))
        optD.load_state_dict(torch.load(load_root / "optD.pt"))

    #######################
    # Create data loaders #
    #######################
    train_set = AudioDataset(
        Path(args.data_path) / "train_files.txt", args.seq_len, sampling_rate=22050
    )
    test_set = AudioDataset(
        Path(args.data_path) / "test_files.txt",
        22050 * 4,
        sampling_rate=22050,
        augment=False,
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=1)

    ##########################
    # Dumping original audio #
    ##########################
    test_voc = []
    test_audio = []
    for i, x_t in enumerate(test_loader):
        x_t = x_t
        s_t = fft(x_t).detach()

        test_voc.append(s_t)
        test_audio.append(x_t)

        audio = x_t.squeeze().cpu()
        save_sample(root / ("original_%d.wav" % i), 22050, audio)
        writer.add_audio("original/sample_%d.wav" % i, audio, 0, sample_rate=22050)

        if i == args.n_test_samples - 1:
            break

    costs = []
    start = time.time()

    # enable cudnn autotuner to speed up training
    torch.backends.cudnn.benchmark = True

    best_mel_reconst = 1000000
    steps = 0
    for epoch in range(1, args.epochs + 1):
        for iterno, x_t in enumerate(train_loader):
            x_t = x_t
            s_t = fft(x_t).detach()
            x_pred_t = netG(s_t)
            with torch.no_grad():
                s_pred_t = fft(x_pred_t.detach())
                s_error = F.l1_loss(s_t, s_pred_t).item()
            #######################
            # Train Discriminator #
            #######################
            D_fake_det = netD(x_pred_t.detach()) # x_pred_t [16,8192] D_fake_det [3*7] 不同尺度特征
            D_real = netD(x_t)  # x_t [16,8192] D_real [3*7] 不同尺度特征
            loss_D = 0
            for scale in D_fake_det:
                loss_D += F.relu(1 + scale[-1]).mean() # len(scale)=7  scale[-1].shape [16,1,32] 希望预测越接近于-1
            for scale in D_real:
                loss_D += F.relu(1 - scale[-1]).mean()  # len(scale)=7 scale[-1].shape [16,1,32] 希望真实越接近于1
            netD.zero_grad()
            loss_D.backward()
            optD.step()

            ###################
            # Train Generator #
            ###################
            D_fake = netD(x_pred_t)  # len(D_fake)=3  len(D_fake[0])=7

            loss_G = 0
            for scale in D_fake:
                loss_G += -scale[-1].mean() # scale[-1].shape [16,1,32] leakyRelu

            loss_feat = 0
            feat_weights = 4.0 / (args.n_layers_D + 1)  # args.n_layers_D = 4  feat_weights=4/5=0.8
            D_weights = 1.0 / args.num_D  # D_weights = 1.0/3=0.33
            wt = D_weights * feat_weights
            for i in range(args.num_D):
                for j in range(len(D_fake[i]) - 1):
                    loss_feat += wt * F.l1_loss(D_fake[i][j], D_real[i][j].detach())

            netG.zero_grad()
            (loss_G + args.lambda_feat * loss_feat).backward()  # args.lambda_feat = 10
            optG.step()

            ######################
            # Update tensorboard #
            ######################
            costs.append([loss_D.item(), loss_G.item(), loss_feat.item(), s_error])

            writer.add_scalar("loss/discriminator", costs[-1][0], steps)
            writer.add_scalar("loss/generator", costs[-1][1], steps)
            writer.add_scalar("loss/feature_matching", costs[-1][2], steps)
            writer.add_scalar("loss/mel_reconstruction", costs[-1][3], steps)
            steps += 1

            if steps % args.save_interval == 0:
                st = time.time()
                with torch.no_grad():
                    for i, (voc, _) in enumerate(zip(test_voc, test_audio)):
                        pred_audio = netG(voc)
                        pred_audio = pred_audio.squeeze().cpu()
                        save_sample(root / ("generated_%d.wav" % i), 22050, pred_audio)
                        writer.add_audio(
                            "generated/sample_%d.wav" % i,
                            pred_audio,
                            epoch,
                            sample_rate=22050,
                        )

                torch.save(netG.state_dict(), root / "netG.pt")
                torch.save(optG.state_dict(), root / "optG.pt")

                torch.save(netD.state_dict(), root / "netD.pt")
                torch.save(optD.state_dict(), root / "optD.pt")

                if np.asarray(costs).mean(0)[-1] < best_mel_reconst:
                    best_mel_reconst = np.asarray(costs).mean(0)[-1]
                    torch.save(netD.state_dict(), root / "best_netD.pt")
                    torch.save(netG.state_dict(), root / "best_netG.pt")

                print("Took %5.4fs to generate samples" % (time.time() - st))
                print("-" * 100)

            if steps % args.log_interval == 0:
                print(
                    "Epoch {} | Iters {} / {} | ms/batch {:5.2f} | loss {}".format(
                        epoch,
                        iterno,
                        len(train_loader),
                        1000 * (time.time() - start) / args.log_interval,
                        np.asarray(costs).mean(0),
                    )
                )
                costs = []
                start = time.time()


if __name__ == "__main__":
    main()
"""
netG:
Generator(
  (model): Sequential(
    (0): ReflectionPad1d((3, 3)) 填补左右边界 以镜像的方式
    (1): Conv1d(80, 512, kernel_size=(7,), stride=(1,))
    (2): LeakyReLU(negative_slope=0.2)
    (3): ConvTranspose1d(512, 256, kernel_size=(16,), stride=(8,), padding=(4,))
    (4): ResnetBlock(
      (block): Sequential(
        (0): LeakyReLU(negative_slope=0.2)
        (1): ReflectionPad1d((1, 1))
        (2): Conv1d(256, 256, kernel_size=(3,), stride=(1,))
        (3): LeakyReLU(negative_slope=0.2)
        (4): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
      )
      (shortcut): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
    )
    (5): ResnetBlock(
      (block): Sequential(
        (0): LeakyReLU(negative_slope=0.2)
        (1): ReflectionPad1d((3, 3))
        (2): Conv1d(256, 256, kernel_size=(3,), stride=(1,), dilation=(3,))
        (3): LeakyReLU(negative_slope=0.2)
        (4): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
      )
      (shortcut): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
    )
    (6): ResnetBlock(
      (block): Sequential(
        (0): LeakyReLU(negative_slope=0.2)
        (1): ReflectionPad1d((9, 9))
        (2): Conv1d(256, 256, kernel_size=(3,), stride=(1,), dilation=(9,))
        (3): LeakyReLU(negative_slope=0.2)
        (4): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
      )
      (shortcut): Conv1d(256, 256, kernel_size=(1,), stride=(1,))
    )
    (7): LeakyReLU(negative_slope=0.2)
    (8): ConvTranspose1d(256, 128, kernel_size=(16,), stride=(8,), padding=(4,))
    (9): ResnetBlock(
      (block): Sequential(
        (0): LeakyReLU(negative_slope=0.2)
        (1): ReflectionPad1d((1, 1))
        (2): Conv1d(128, 128, kernel_size=(3,), stride=(1,))
        (3): LeakyReLU(negative_slope=0.2)
        (4): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
      )
      (shortcut): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
    )
    (10): ResnetBlock(
      (block): Sequential(
        (0): LeakyReLU(negative_slope=0.2)
        (1): ReflectionPad1d((3, 3))
        (2): Conv1d(128, 128, kernel_size=(3,), stride=(1,), dilation=(3,))
        (3): LeakyReLU(negative_slope=0.2)
        (4): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
      )
      (shortcut): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
    )
    (11): ResnetBlock(
      (block): Sequential(
        (0): LeakyReLU(negative_slope=0.2)
        (1): ReflectionPad1d((9, 9))
        (2): Conv1d(128, 128, kernel_size=(3,), stride=(1,), dilation=(9,))
        (3): LeakyReLU(negative_slope=0.2)
        (4): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
      )
      (shortcut): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
    )
    (12): LeakyReLU(negative_slope=0.2)
    (13): ConvTranspose1d(128, 64, kernel_size=(4,), stride=(2,), padding=(1,))
    (14): ResnetBlock(
      (block): Sequential(
        (0): LeakyReLU(negative_slope=0.2)
        (1): ReflectionPad1d((1, 1))
        (2): Conv1d(64, 64, kernel_size=(3,), stride=(1,))
        (3): LeakyReLU(negative_slope=0.2)
        (4): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
      )
      (shortcut): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
    )
    (15): ResnetBlock(
      (block): Sequential(
        (0): LeakyReLU(negative_slope=0.2)
        (1): ReflectionPad1d((3, 3))
        (2): Conv1d(64, 64, kernel_size=(3,), stride=(1,), dilation=(3,))
        (3): LeakyReLU(negative_slope=0.2)
        (4): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
      )
      (shortcut): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
    )
    (16): ResnetBlock(
      (block): Sequential(
        (0): LeakyReLU(negative_slope=0.2)
        (1): ReflectionPad1d((9, 9))
        (2): Conv1d(64, 64, kernel_size=(3,), stride=(1,), dilation=(9,))
        (3): LeakyReLU(negative_slope=0.2)
        (4): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
      )
      (shortcut): Conv1d(64, 64, kernel_size=(1,), stride=(1,))
    )
    (17): LeakyReLU(negative_slope=0.2)
    (18): ConvTranspose1d(64, 32, kernel_size=(4,), stride=(2,), padding=(1,))
    (19): ResnetBlock(
      (block): Sequential(
        (0): LeakyReLU(negative_slope=0.2)
        (1): ReflectionPad1d((1, 1))
        (2): Conv1d(32, 32, kernel_size=(3,), stride=(1,))
        (3): LeakyReLU(negative_slope=0.2)
        (4): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
      )
      (shortcut): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
    )
    (20): ResnetBlock(
      (block): Sequential(
        (0): LeakyReLU(negative_slope=0.2)
        (1): ReflectionPad1d((3, 3))
        (2): Conv1d(32, 32, kernel_size=(3,), stride=(1,), dilation=(3,))
        (3): LeakyReLU(negative_slope=0.2)
        (4): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
      )
      (shortcut): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
    )
    (21): ResnetBlock(
      (block): Sequential(
        (0): LeakyReLU(negative_slope=0.2)
        (1): ReflectionPad1d((9, 9))
        (2): Conv1d(32, 32, kernel_size=(3,), stride=(1,), dilation=(9,))
        (3): LeakyReLU(negative_slope=0.2)
        (4): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
      )
      (shortcut): Conv1d(32, 32, kernel_size=(1,), stride=(1,))
    )
    (22): LeakyReLU(negative_slope=0.2)
    (23): ReflectionPad1d((3, 3))
    (24): Conv1d(32, 1, kernel_size=(7,), stride=(1,))
    (25): Tanh()
  )
)
"""


"""
netD:
Discriminator(
  (model): ModuleDict(
    (disc_0): NLayerDiscriminator(
      (model): ModuleDict(
        (layer_0): Sequential(
          (0): ReflectionPad1d((7, 7))
          (1): Conv1d(1, 16, kernel_size=(15,), stride=(1,))
          (2): LeakyReLU(negative_slope=0.2, inplace=True)
        )
        (layer_1): Sequential(
          (0): Conv1d(16, 64, kernel_size=(41,), stride=(4,), padding=(20,), groups=4)
          (1): LeakyReLU(negative_slope=0.2, inplace=True)
        )
        (layer_2): Sequential(
          (0): Conv1d(64, 256, kernel_size=(41,), stride=(4,), padding=(20,), groups=16)
          (1): LeakyReLU(negative_slope=0.2, inplace=True)
        )
        (layer_3): Sequential(
          (0): Conv1d(256, 1024, kernel_size=(41,), stride=(4,), padding=(20,), groups=64)
          (1): LeakyReLU(negative_slope=0.2, inplace=True)
        )
        (layer_4): Sequential(
          (0): Conv1d(1024, 1024, kernel_size=(41,), stride=(4,), padding=(20,), groups=256)
          (1): LeakyReLU(negative_slope=0.2, inplace=True)
        )
        (layer_5): Sequential(
          (0): Conv1d(1024, 1024, kernel_size=(5,), stride=(1,), padding=(2,))
          (1): LeakyReLU(negative_slope=0.2, inplace=True)
        )
        (layer_6): Conv1d(1024, 1, kernel_size=(3,), stride=(1,), padding=(1,))
      )
    )
    (disc_1): NLayerDiscriminator(
      (model): ModuleDict(
        (layer_0): Sequential(
          (0): ReflectionPad1d((7, 7))
          (1): Conv1d(1, 16, kernel_size=(15,), stride=(1,))
          (2): LeakyReLU(negative_slope=0.2, inplace=True)
        )
        (layer_1): Sequential(
          (0): Conv1d(16, 64, kernel_size=(41,), stride=(4,), padding=(20,), groups=4)
          (1): LeakyReLU(negative_slope=0.2, inplace=True)
        )
        (layer_2): Sequential(
          (0): Conv1d(64, 256, kernel_size=(41,), stride=(4,), padding=(20,), groups=16)
          (1): LeakyReLU(negative_slope=0.2, inplace=True)
        )
        (layer_3): Sequential(
          (0): Conv1d(256, 1024, kernel_size=(41,), stride=(4,), padding=(20,), groups=64)
          (1): LeakyReLU(negative_slope=0.2, inplace=True)
        )
        (layer_4): Sequential(
          (0): Conv1d(1024, 1024, kernel_size=(41,), stride=(4,), padding=(20,), groups=256)
          (1): LeakyReLU(negative_slope=0.2, inplace=True)
        )
        (layer_5): Sequential(
          (0): Conv1d(1024, 1024, kernel_size=(5,), stride=(1,), padding=(2,))
          (1): LeakyReLU(negative_slope=0.2, inplace=True)
        )
        (layer_6): Conv1d(1024, 1, kernel_size=(3,), stride=(1,), padding=(1,))
      )
    )
    (disc_2): NLayerDiscriminator(
      (model): ModuleDict(
        (layer_0): Sequential(
          (0): ReflectionPad1d((7, 7))
          (1): Conv1d(1, 16, kernel_size=(15,), stride=(1,))
          (2): LeakyReLU(negative_slope=0.2, inplace=True)
        )
        (layer_1): Sequential(
          (0): Conv1d(16, 64, kernel_size=(41,), stride=(4,), padding=(20,), groups=4)
          (1): LeakyReLU(negative_slope=0.2, inplace=True)
        )
        (layer_2): Sequential(
          (0): Conv1d(64, 256, kernel_size=(41,), stride=(4,), padding=(20,), groups=16)
          (1): LeakyReLU(negative_slope=0.2, inplace=True)
        )
        (layer_3): Sequential(
          (0): Conv1d(256, 1024, kernel_size=(41,), stride=(4,), padding=(20,), groups=64)
          (1): LeakyReLU(negative_slope=0.2, inplace=True)
        )
        (layer_4): Sequential(
          (0): Conv1d(1024, 1024, kernel_size=(41,), stride=(4,), padding=(20,), groups=256)
          (1): LeakyReLU(negative_slope=0.2, inplace=True)
        )
        (layer_5): Sequential(
          (0): Conv1d(1024, 1024, kernel_size=(5,), stride=(1,), padding=(2,))
          (1): LeakyReLU(negative_slope=0.2, inplace=True)
        )
        (layer_6): Conv1d(1024, 1, kernel_size=(3,), stride=(1,), padding=(1,))
      )
    )
  )
  (downsample): AvgPool1d(kernel_size=(4,), stride=(2,), padding=(1,))
)
"""