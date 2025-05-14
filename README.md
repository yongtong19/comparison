# ⚠️ Caution

Typically, you need to modify `loader.py` to load your own dataset.

The current `loader.py` is designed for following dataset structure:

```
dataset/
    img/
        img_1.mat
        img_2.mat
        ...
    label/
        label_1.mat
        label_2.mat
        ...
```

# Clone the repository

```bash
git clone https://github.com/yongtong19/comparison.git
```

# Create a virtual environment

## Option 1 (with venv and pip)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Option 2 (with uv, recommended)

[`uv`](https://github.com/astral-sh/uv?tab=readme-ov-file#uv) is a extremely fast Python package and project manager.
You can install `uv` by:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

or install with snap on Ubuntu:

```bash
sudo snap install uv --classic
```

After installing `uv`, you can install the dependencies by:

```bash
uv sync --locked
```

# Training

Simplest way:

```bash
python main.py --model_name MODEL_NAME --dataset DATASET
```

or with uv:

```bash
uv run python main.py --model_name MODEL_NAME --dataset DATASET
```

By default, the checkpoints of each epoch will be saved as `./save/[datetime]/checkpoints/[model_name]_[epoch].ckpt`, the corresponding configurations will be saved under `save/[datetime]/config.yaml`.

# Testing

Simplest way:

```bash
python main.py --model_name MODEL_NAME --dataset TEST_DATASET --mode test --test_checkpoint_path CHECKPOINT_PATH
```

or with uv:

```bash
uv run python main.py --model_name MODEL_NAME --dataset TEST_DATASET --mode test --save_path SAVE_PATH
```

By default, the latest checkpoint under `save_path` will be used for testing. SSIM and PSNR of each testing sample will be saved in `save_path/test_result/metrics.csv`. Average SSIM and PSNR will be printed and saved in `save_path/test_result/average_metrics.txt`. The output of each testing sample will also be saved under `save_path/test_result/predictions/` if `--test_save_result` is set to `True`.

# Help

```
Usage: main.py [OPTIONS]

Options:
  --mode TEXT                     train or test, default 'train'
  --model_name TEXT               model name, default 'REDCNN'
  --batch_size INTEGER            batch size, default 1
  --dataset TEXT                  the path of dataset  [required]
  --save_path TEXT                For training, checkpoints will be saved
                                  under this path. For testing, checkpoints
                                  will be loaded from this path and testing
                                  results will also be saved under it.
  --image_size INTEGER            image size, default 512
  --device TEXT                   training device, default 'cuda'
  --num_workers INTEGER           number of data loading processes, default 4
  --train_num_epochs INTEGER      number of training epochs, default 20
  --train_log_interval INTEGER    number of training log interval, default 20
  --train_decay_interval INTEGER  number of training learning rate decay
                                  interval, default 1000
  --train_lr FLOAT                initial training learning rate, default 1e-3
  --train_criterion TEXT          training criterion, support l1, l2/mse,
                                  default 'mse'
  --train_use_amp BOOLEAN         enable mixed precision training, default
                                  True
  --test_data_range FLOAT         psnr and ssim data range during testing,
                                  default 1.0
  --test_save_result BOOLEAN      save testing results, if set to False, the
                                  results will not be saved, default True
  --test_checkpoint_path TEXT     testing checkpoint path, if not given, the
                                  latest checkpoint under save_path will be
                                  used, default None
  --help                          Show this message and exit.
```
