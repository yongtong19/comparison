import os
import time
import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast
from torch.utils.data import DataLoader

import models
from metrics import compute_metrics
from utils import printProgressBar


class Solver(object):
    def __init__(
        self,
        config: dict,
        data_loader: DataLoader,
    ):
        self.mode = config["mode"]
        self.data_loader: DataLoader = data_loader
        self.image_size = (
            data_loader.dataset.image_size
            if hasattr(data_loader.dataset, "image_size")
            else config["image_size"]
        )

        if config["device"]:
            self.device = torch.device(config["device"])
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Add mixed precision settings
        self.train_use_amp = config["train_use_amp"]
        self.dtype = torch.bfloat16 if config["train_use_amp"] else torch.float32

        self.save_path = config["save_path"]
        self.batch_size = config["batch_size"]
        self.train_num_epochs = config["train_num_epochs"]
        self.train_log_interval = config["train_log_interval"]
        self.train_decay_interval = config["train_decay_interval"]
        self.train_lr = config["train_lr"]
        self.test_checkpoint_path = config["test_checkpoint_path"]
        self.test_data_range = config["test_data_range"]
        self.test_save_result = config["test_save_result"]

        self.model_name = config["model_name"]
        if (self.model_name).lower() == "redcnn":
            self.model = models.redcnn.RED_CNN()
        elif (self.model_name).lower() == "swinir":
            self.model = models.swinir.SwinIR()
        elif (self.model_name).lower() == "restormer":
            self.model = models.restormer.Restormer()
        elif (self.model_name).lower() == "unet":
            self.model = models.unet.UNet()

        self.model.to(self.device)

        if (
            config["train_criterion"].lower() == "mse"
            or config["train_criterion"].lower() == "l2"
        ):
            self.criterion = nn.MSELoss()
        elif config["train_criterion"].lower() == "l1":
            self.criterion = nn.L1Loss()

        self.optimizer = optim.Adam(self.model.parameters(), self.train_lr)

    def find_checkpoint(self):
        if self.test_checkpoint_path:
            return self.test_checkpoint_path
        else:
            # find the latest checkpoint under save_path by modified time
            checkpoints = os.listdir(os.path.join(self.save_path, "checkpoints"))
            checkpoints = [
                os.path.join(self.save_path, "checkpoints", checkpoint)
                for checkpoint in checkpoints
            ]
            return max(checkpoints, key=os.path.getmtime)

    def save_model(self, epoch):
        os.makedirs(os.path.join(self.save_path, "checkpoints"), exist_ok=True)
        checkpoint_path = os.path.join(
            self.save_path,
            "checkpoints",
            f"{self.model_name}_{epoch}.ckpt",
        )
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"checkpoint saved: {checkpoint_path}")

    def load_model(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))

    def lr_decay(self, current_epoch):
        min_lr = 1e-6
        max_lr = self.train_lr
        total_epochs = self.train_num_epochs

        # Cosine annealing schedule
        lr = min_lr + 0.5 * (max_lr - min_lr) * (
            1 + np.cos(np.pi * current_epoch / total_epochs)
        )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def train(self):
        cumulated_psnr = 0
        cumulated_ssim = 0
        cumulated_steps = 0
        start_time = time.time()
        for epoch in range(1, self.train_num_epochs + 1):
            self.model.train(True)

            for step, (input_img, target_img) in enumerate(self.data_loader, 1):
                cumulated_steps += 1
                input_img = input_img.to(self.device, dtype=self.dtype)
                target_img = target_img.to(self.device, dtype=self.dtype)

                self.optimizer.zero_grad()

                with autocast(
                    device_type=self.device.type,
                    dtype=self.dtype,
                    enabled=self.train_use_amp,
                ):
                    prediction = self.model(input_img)
                    loss: torch.Tensor = self.criterion(prediction, target_img)

                loss.backward()
                self.optimizer.step()

                data_range = 1
                pred_result = compute_metrics(
                    target_img,
                    prediction,
                    data_range,
                )

                cumulated_psnr += pred_result[0]
                cumulated_ssim += pred_result[1]

                # log training progress
                if cumulated_steps % self.train_log_interval == 0:
                    print(
                        "[{}] EPOCH [{}/{}], STEP [{}/{}] LOSS: {:.8f}, TIME: {:.1f}s, AVG PSNR: {:.3f}, AVG SSIM: {:.3f}".format(
                            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            epoch,
                            self.train_num_epochs,
                            step,
                            len(self.data_loader),
                            loss.item(),
                            time.time() - start_time,
                            cumulated_psnr / cumulated_steps,
                            cumulated_ssim / cumulated_steps,
                        )
                    )

                # learning rate decay
                if (
                    self.train_decay_interval != 0
                    and cumulated_steps % self.train_decay_interval == 0
                ):
                    self.lr_decay(epoch)

            self.save_model(epoch)

    def test(self):
        metrics = []
        os.makedirs(os.path.join(self.save_path, "test_result"), exist_ok=True)
        if self.test_save_result:
            os.makedirs(
                os.path.join(self.save_path, "test_result", "predictions"),
                exist_ok=True,
            )

        with torch.no_grad():
            self.load_model(self.test_checkpoint_path)
            self.model.train(False)

            for step, (input_img, target_img) in enumerate(self.data_loader):
                input_img = input_img.float().to(self.device)
                target_img = target_img.float().to(self.device)
                prediction = self.model(input_img)

                data_range = self.test_data_range
                pred_result = compute_metrics(
                    prediction,
                    target_img,
                    data_range,
                )

                metrics.append(pred_result)
                # save result figure
                if self.test_save_result:
                    np.save(
                        os.path.join(
                            self.save_path, "test_result", "predictions", f"{step}.npy"
                        ),
                        prediction.cpu().numpy(),
                    )

                printProgressBar(
                    step,
                    len(self.data_loader),
                    prefix="Compute measurements ..",
                    suffix="Complete",
                    length=25,
                )

            metrics = pd.DataFrame(metrics, columns=["PSNR", "SSIM"])
            metrics.to_csv(
                os.path.join(self.save_path, "test_result", "metrics.csv"),
                index=False,
            )
            print(
                f"Average PSNR: {metrics['PSNR'].mean()}, Average SSIM: {metrics['SSIM'].mean()}"
            )
            with open(
                os.path.join(self.save_path, "test_result", "average_metrics.txt"), "w"
            ) as f:
                f.write(
                    f"Average PSNR: {metrics['PSNR'].mean()}, Average SSIM: {metrics['SSIM'].mean()}"
                )
