import math

import torch
from torch import nn, Tensor
from torchvision.transforms import autoaugment as TA, functional as TF
from torchvision.transforms.autoaugment import _apply_op
from .utils import transformation_forward_wrapper

class AutoAugment(TA.AutoAugment):
    @transformation_forward_wrapper
    def forward(self, img):
        transform_id, probs, signs = self.get_params(len(self.policies))

        out_imgs = []
        for i in img:
            fill = self.fill
            if isinstance(i, Tensor):
                if isinstance(fill, (int, float)):
                    fill = [float(fill)] * TF.get_image_num_channels(i)
                elif fill is not None:
                    fill = [float(f) for f in fill]

            for i, (op_name, p, magnitude_id) in enumerate(self.policies[transform_id]):
                if probs[i] <= p:
                    op_meta = self._augmentation_space(10, TF.get_image_size(i))
                    magnitudes, signed = op_meta[op_name]
                    magnitude = float(magnitudes[magnitude_id].item()) if magnitude_id is not None else 0.0
                    if signed and signs[i] == 0:
                        magnitude *= -1.0
                    i = _apply_op(i, op_name, magnitude, interpolation=self.interpolation, fill=fill)
            out_imgs.append(i)
        return out_imgs


class RandAugment(TA.RandAugment):
    @transformation_forward_wrapper
    def forward(self, img):
        out_imgs = []

        random_state = torch.random.get_rng_state()
        for i in img:
            torch.random.set_rng_state(random_state)

            fill = self.fill
            if isinstance(i, Tensor):
                if isinstance(fill, (int, float)):
                    fill = [float(fill)] * TF.get_image_num_channels(i)
                elif fill is not None:
                    fill = [float(f) for f in fill]

            for _ in range(self.num_ops):
                op_meta = self._augmentation_space(self.num_magnitude_bins, TF.get_image_size(i))
                op_index = int(torch.randint(len(op_meta), (1,)).item())
                op_name = list(op_meta.keys())[op_index]
                magnitudes, signed = op_meta[op_name]
                magnitude = float(magnitudes[self.magnitude].item()) if magnitudes.ndim > 0 else 0.0
                if signed and torch.randint(2, (1,)):
                    magnitude *= -1.0
                i = _apply_op(i, op_name, magnitude, interpolation=self.interpolation, fill=fill)
            out_imgs.append(i)
        return out_imgs


class TrivialAugmentWide(TA.TrivialAugmentWide):
    @transformation_forward_wrapper
    def forward(self, img):
        out_imgs = []

        random_state = torch.random.get_rng_state()
        for i in img:
            torch.random.set_rng_state(random_state)

            fill = self.fill
            if isinstance(i, Tensor):
                if isinstance(fill, (int, float)):
                    fill = [float(fill)] * TF.get_image_num_channels(i)
                elif fill is not None:
                    fill = [float(f) for f in fill]

            op_meta = self._augmentation_space(self.num_magnitude_bins)
            op_index = int(torch.randint(len(op_meta), (1,)).item())
            op_name = list(op_meta.keys())[op_index]
            magnitudes, signed = op_meta[op_name]
            magnitude = (
                float(magnitudes[torch.randint(len(magnitudes), (1,), dtype=torch.long)].item())
                if magnitudes.ndim > 0
                else 0.0
            )
            if signed and torch.randint(2, (1,)):
                magnitude *= -1.0
            i = _apply_op(i, op_name, magnitude, interpolation=self.interpolation, fill=fill)
            out_imgs.append(i)
        return out_imgs


class RandomMixup(nn.Module):
    def __init__(self, p: float = 0.5, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        assert alpha > 0, "Alpha param can't be zero."

        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch_image, batch_mask):
        if batch_image.ndim != 4 or batch_mask.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch_image.ndim} and {batch_mask.ndim}")
        if not batch_image.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch_image.dtype}.")

        if not self.inplace:
            batch_image = batch_image.clone()
            batch_mask = batch_mask.clone()

        if torch.rand(1).item() >= self.p:
            return batch_image, batch_mask

        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])

        batch_rolled_image = batch_image.roll(1, 0)
        batch_rolled_image.mul_(1.0 - lambda_param)
        batch_image.mul_(lambda_param).add_(batch_rolled_image)

        batch_rolled_mask = batch_mask.roll(1, 0)
        batch_rolled_mask.mul_(1.0 - lambda_param)
        batch_mask.mul_(lambda_param).add_(batch_rolled_mask)

        return batch_image, batch_mask

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "p={p}"
        s += ", alpha={alpha}"
        s += ", inplace={inplace}"
        s += ")"
        return s.format(**self.__dict__)


class RandomCutmix(nn.Module):
    def __init__(self, p: float = 0.5, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        assert alpha > 0, "Alpha param can't be zero."

        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch_image, batch_mask):
        if batch_image.ndim != 4 or batch_mask.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch_image.ndim} and {batch_mask.ndim}")
        if not batch_image.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch_image.dtype}.")

        if not self.inplace:
            batch_image = batch_image.clone()
            batch_mask = batch_mask.clone()

        if torch.rand(1).item() >= self.p:
            return batch_image, batch_mask

        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        W, H = TF.get_image_size(batch_image)

        r_x = torch.randint(W, (1,))
        r_y = torch.randint(H, (1,))

        r = 0.5 * math.sqrt(1.0 - lambda_param)
        r_w_half = int(r * W)
        r_h_half = int(r * H)

        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled_image = batch_image.roll(1, 0)
        batch_image[:, :, y1:y2, x1:x2] = batch_rolled_image[:, :, y1:y2, x1:x2]
        batch_rolled_mask = batch_mask.roll(1, 0)
        batch_mask[:, :, y1:y2, x1:x2] = batch_rolled_mask[:, :, y1:y2, x1:x2]
        return batch_image, batch_mask

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "p={p}"
        s += ", alpha={alpha}"
        s += ", inplace={inplace}"
        s += ")"
        return s.format(**self.__dict__)
