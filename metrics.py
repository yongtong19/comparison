from torchmetrics.functional.image import (
    peak_signal_noise_ratio as psnr,
    structural_similarity_index_measure as ssim,
)


def compute_metrics(y, pred, data_range):
    pred_psnr = psnr(pred, y, data_range=data_range).item()
    pred_ssim = ssim(pred, y).item()
    return pred_psnr, pred_ssim
