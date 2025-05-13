import os
import argparse
from torch.backends import cudnn
from loader import get_loader
from solver import Solver
from datetime import datetime
import yaml


def dump_config(args):
    with open(os.path.join(args.save_path, "config.yaml"), "w") as f:
        yaml.dump(args, f)


def main(args):
    cudnn.benchmark = True

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print("Create path : {}".format(args.save_path))

    if args.result_fig:
        fig_path = os.path.join(args.save_path, "fig")
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
            print("Create path : {}".format(fig_path))

    data_loader = get_loader(
        mode=args.mode,
        saved_path=args.saved_path,
        dataset=args.dataset,
        batch_size=(args.batch_size if args.mode == "train" else 1),
        shuffle=(True if args.mode == "train" else False),
        num_workers=args.num_workers,
    )

    solver = Solver(args, data_loader)
    dump_config(args)

    if args.mode == "train":
        solver.train()
    elif args.mode == "test":
        solver.test()
    elif args.mode == "test2":
        solver.single_test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--data_path", type=str, default="./AAPM-Mayo-CT-Challenge/")
    parser.add_argument(
        "--saved_path",
        type=str,
        default="/mnt/4b9cdae1-f581-4f95-aa23-5b45c0bdf521/wday/aapm_all_npy/aapm_all_npy/",
    )  ## use 3mm data
    parser.add_argument(
        "--save_path",
        type=str,
        default=f"save/{datetime.now().strftime('%Y%m%d%H%M%S')}",
    )
    parser.add_argument("--result_fig", type=bool, default=True)

    parser.add_argument("--num_epochs", type=int, default=20)  ## 200 or 2000
    parser.add_argument("--print_iters", type=int, default=20)
    parser.add_argument("--decay_iters", type=int, default=3000)
    parser.add_argument(
        "--save_iters", type=int, default=1500
    )  ## the iterats~epochs*10 useless for now
    parser.add_argument("--test_iters", type=int, default=40379)

    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--device", type=str)
    parser.add_argument("--num_workers", type=int, default=7)

    parser.add_argument("--model_name", type=str, default="REDCNN")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--test_data_range", type=float, default=1.0)
    args = parser.parse_args()
    main(args)
