import os
import time
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import lpips

from measure import compute_measure
from prep import printProgressBar

from swinir import SwinIR
from redcnn import RED_CNN
from restormer import Restormer
from torch.utils.data import DataLoader


class Solver(object):
    def __init__(self, args, data_loader: DataLoader):
        self.mode = args.mode
        self.data_loader = data_loader
        self.image_size = data_loader.dataset.image_size

        if args.device:
            self.device = torch.device(args.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.save_path = args.save_path
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.print_iters = args.print_iters
        self.decay_iters = args.decay_iters
        self.save_iters = args.save_iters
        self.test_iters = args.test_iters
        self.test_data_range = args.test_data_range
        self.result_fig = args.result_fig

        self.model_name = args.model_name
        if (self.model_name).lower() == "redcnn":
            self.model = RED_CNN()
        elif (self.model_name).lower() == "swinir":
            self.model = SwinIR()
        elif (self.model_name).lower() == "restormer":
            self.model = Restormer()

        self.model.to(self.device)

        self.lr = args.lr
        self.criterion = nn.MSELoss()

        self.optimizer = optim.Adam(self.model.parameters(), self.lr)
        self.lpips_fn = lpips.LPIPS(net="alex").to(self.device)

    def save_model(self, iter_):
        f = os.path.join(
            self.save_path, "{}_{}iter.ckpt".format(self.model_name, iter_)
        )
        torch.save(self.model.state_dict(), f)

    def load_model(self, iter_):
        f = os.path.join(
            self.save_path, "{}_{}iter.ckpt".format(self.model_name, iter_)
        )
        self.model.load_state_dict(torch.load(f))

    def lr_decay(self):
        lr = self.lr * 0.5
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def save_fig(self, x, y, pred, fig_name, original_result, pred_result):
        x = (x - x.min()) / (x.max() - x.min())
        x, y, pred = (
            x.view(self.image_size, self.image_size).detach().cpu().numpy(),
            y.view(self.image_size, self.image_size).detach().cpu().numpy(),
            pred.view(self.image_size, self.image_size).detach().cpu().numpy(),
        )
        f, ax = plt.subplots(1, 3, figsize=(30, 10))
        vmin1 = 0
        vmax1 = 1
        ax[0].imshow(x, cmap=plt.cm.gray)
        ax[0].set_title("Input", fontsize=30)
        ax[0].set_xlabel(
            "PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}\nLPIPS: {:.4f}".format(
                original_result[0],
                original_result[1],
                original_result[2],
                original_result[3],
            ),
            fontsize=20,
        )
        ax[1].imshow(pred, cmap=plt.cm.gray, vmin=vmin1, vmax=vmax1)
        ax[1].set_title("Prediction", fontsize=30)
        ax[1].set_xlabel(
            "PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}\nLPIPS: {:.4f}".format(
                pred_result[0], pred_result[1], pred_result[2], pred_result[3]
            ),
            fontsize=20,
        )
        ax[2].imshow(y, cmap=plt.cm.gray, vmin=vmin1, vmax=vmax1)
        ax[2].set_title("Target", fontsize=30)

        f.savefig(os.path.join(self.save_path, "fig", "result_{}.png".format(fig_name)))
        plt.close()

    def train(self):
        total_psnr = 0
        total_ssim = 0
        total_lpips = 0
        train_losses = []
        total_iters = 0
        start_time = time.time()
        for epoch in range(1, self.num_epochs + 1):
            self.model.train(True)

            for iter_, (input_img, target_img, _, _) in enumerate(self.data_loader):
                total_iters += 1

                input_img = input_img.float().to(self.device)
                target_img = target_img.float().to(self.device)
                prediction = self.model(input_img)

                data_range = 1
                original_result, pred_result = compute_measure(
                    input_img,
                    target_img,
                    prediction,
                    data_range,
                    self.lpips_fn,
                    self.image_size,
                )

                total_psnr += pred_result[0]
                total_ssim += pred_result[1]
                total_lpips += pred_result[3]
                loss = self.criterion(prediction, target_img)
                self.model.zero_grad()
                self.optimizer.zero_grad()

                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())

                # print
                if total_iters % self.print_iters == 0:
                    print(
                        "STEP [{}], EPOCH [{}/{}], ITER [{}/{}] \nLOSS: {:.8f}, TIME: {:.1f}s, AVG SSIM: {:.3f}, AVG PSNR: {:.2f}, AVG LPIPS: {:.4f}".format(
                            total_iters,
                            epoch,
                            self.num_epochs,
                            iter_ + 1,
                            len(self.data_loader),
                            loss.item(),
                            time.time() - start_time,
                            total_ssim / total_iters,
                            total_psnr / total_iters,
                            total_lpips / total_iters,
                        )
                    )
                # if total_iters % self.decay_iters == 0:
                #    self.lr_decay()
                # save model
                if total_iters % 5000 == 0:
                    print("save model: true")
                    self.save_model(total_iters)
                    np.save(
                        os.path.join(
                            self.save_path, "loss_{}_iter.npy".format(total_iters)
                        ),
                        np.array(train_losses),
                    )

                if total_iters % 1000 == 0:
                    self.save_fig(
                        input_img,
                        target_img,
                        prediction,
                        iter_ + 1,
                        original_result,
                        pred_result,
                    )

        self.save_model(total_iters)
        np.save(
            os.path.join(self.save_path, "loss_{}_iter.npy".format(total_iters)),
            np.array(train_losses),
        )
        print("total_iters:", total_iters)

    def test(self):
        total_psnr = 0
        total_ssim = 0
        total_lpips = 0

        total_iters = 0
        # compute PSNR, SSIM, RMSE
        ori_psnr_avg, ori_ssim_avg, ori_rmse_avg, ori_lpips_avg = 0, 0, 0, 0
        pred_psnr_avg, pred_ssim_avg, pred_rmse_avg, pred_lpips_avg = 0, 0, 0, 0
        with torch.no_grad():
            self.load_model(self.test_iters)
            self.model.train(False)

            for iter_, (input_img, target_img, input_files, target_files) in enumerate(
                self.data_loader
            ):
                total_iters += 1

                input_img = input_img.float().to(self.device)
                target_img = target_img.float().to(self.device)
                prediction = self.model(input_img)

                data_range = self.test_data_range
                original_result, pred_result = compute_measure(
                    input_img,
                    target_img,
                    prediction,
                    data_range,
                    self.lpips_fn,
                    self.image_size,
                )

                total_psnr += pred_result[0]
                total_ssim += pred_result[1]
                total_lpips += pred_result[3]
                ori_psnr_avg += original_result[0]
                ori_ssim_avg += original_result[1]
                ori_rmse_avg += original_result[2]
                ori_lpips_avg += original_result[3]
                pred_psnr_avg += pred_result[0]
                pred_ssim_avg += pred_result[1]
                pred_rmse_avg += pred_result[2]
                pred_lpips_avg += pred_result[3]

                # save result figure
                if self.result_fig:
                    self.save_fig(
                        input_img,
                        target_img,
                        prediction,
                        iter_,
                        original_result,
                        pred_result,
                    )
                    os.makedirs(
                        os.path.join(self.save_path, "test_result"), exist_ok=True
                    )
                    np.save(
                        os.path.join(self.save_path, "test_result", input_files[0]),
                        prediction.cpu().numpy(),
                    )

                printProgressBar(
                    iter_,
                    len(self.data_loader),
                    prefix="Compute measurements ..",
                    suffix="Complete",
                    length=25,
                )
            print("\n")
            print(
                "Original === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f} \nLPIPS avg: {:.4f}".format(
                    ori_psnr_avg / len(self.data_loader),
                    ori_ssim_avg / len(self.data_loader),
                    ori_rmse_avg / len(self.data_loader),
                    ori_lpips_avg / len(self.data_loader),
                )
            )
            print("\n")
            print(
                "Predictions === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f} \nLPIPS avg: {:.4f}".format(
                    pred_psnr_avg / len(self.data_loader),
                    pred_ssim_avg / len(self.data_loader),
                    pred_rmse_avg / len(self.data_loader),
                    pred_lpips_avg / len(self.data_loader),
                )
            )

        print(
            "TEST STEP [{}], AVG SSIM: {:.3f}, AVG PSNR: {:.2f}, AVG LPIPS: {:.4f}".format(
                total_iters,
                total_ssim / total_iters,
                total_psnr / total_iters,
                total_lpips / total_iters,
            )
        )
