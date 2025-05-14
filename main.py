import os
from loader import get_loader
from solver import Solver
from datetime import datetime
import yaml
import click


@click.command()
@click.option(
    "--mode", type=str, default="train", help="train or test, default 'train'"
)
@click.option(
    "--model_name", type=str, default="REDCNN", help="model name, default 'REDCNN'"
)
@click.option("--batch_size", type=int, default=1, help="batch size, default 1")
@click.option("--dataset", required=True, type=str, help="the path of dataset")
@click.option(
    "--save_path",
    type=str,
    default=f"save/{datetime.now().strftime('%Y%m%d%H%M%S')}",
    help="""
For training, checkpoints will be saved under this path.
For testing, checkpoints will be loaded from this path and testing results will also be saved under it.
""",
)
@click.option("--image_size", type=int, default=512, help="image size, default 512")
@click.option(
    "--device", type=str, default="cuda", help="training device, default 'cuda'"
)
@click.option(
    "--num_workers",
    type=int,
    default=4,
    help="number of data loading processes, default 4",
)
@click.option(
    "--train_num_epochs",
    type=int,
    default=20,
    help="number of training epochs, default 20",
)
@click.option(
    "--train_log_interval",
    type=int,
    default=20,
    help="number of training log interval, default 20",
)
@click.option(
    "--train_decay_interval",
    type=int,
    default=1000,
    help="number of training learning rate decay interval, default 1000",
)
@click.option(
    "--train_lr",
    type=float,
    default=1e-3,
    help="initial training learning rate, default 1e-3",
)
@click.option(
    "--train_criterion",
    type=str,
    default="mse",
    help="training criterion, support l1, l2/mse, default 'mse'",
)
@click.option(
    "--train_use_amp",
    type=bool,
    default=True,
    help="enable mixed precision training, default True",
)
@click.option(
    "--test_data_range",
    type=float,
    default=1.0,
    help="psnr and ssim data range during testing, default 1.0",
)
@click.option(
    "--test_save_result",
    type=bool,
    default=True,
    help="save testing results, if set to False, the results will not be saved, default True",
)
@click.option(
    "--test_checkpoint_path",
    type=str,
    default=None,
    help="testing checkpoint path, if not given, the latest checkpoint under save_path will be used, default None",
)
def main(
    mode,
    model_name,
    batch_size,
    dataset,
    save_path,
    image_size,
    device,
    num_workers,
    train_num_epochs,
    train_log_interval,
    train_decay_interval,
    train_lr,
    train_criterion,
    train_use_amp,
    test_data_range,
    test_save_result,
    test_checkpoint_path,
):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print("Create path : {}".format(save_path))

    solver_config = {
        "mode": mode,
        "model_name": model_name,
        "batch_size": batch_size,
        "dataset": dataset,
        "save_path": save_path,
        "image_size": image_size,
        "device": device,
        "num_workers": num_workers,
        "train_num_epochs": train_num_epochs,
        "train_log_interval": train_log_interval,
        "train_decay_interval": train_decay_interval,
        "train_lr": train_lr,
        "train_criterion": train_criterion,
        "train_use_amp": train_use_amp,
        "test_data_range": test_data_range,
        "test_save_result": test_save_result,
        "test_checkpoint_path": test_checkpoint_path,
    }

    with open(os.path.join(save_path, "config.yaml"), "w") as f:
        yaml.dump(solver_config, f)

    data_loader = get_loader(
        dataset=dataset,
        batch_size=(batch_size if mode == "train" else 1),
        shuffle=(True if mode == "train" else False),
        num_workers=num_workers,
    )

    solver = Solver(
        config=solver_config,
        data_loader=data_loader,
    )

    if mode == "train":
        solver.train()
    elif mode == "test":
        solver.test()


if __name__ == "__main__":
    main()
