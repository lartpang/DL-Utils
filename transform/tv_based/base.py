import torch
from torch import Tensor
from torchvision.transforms import functional as TF, transforms as TT

from .utils import transformation_forward_wrapper


class RandomHorizontalFlip(TT.RandomHorizontalFlip):
    @transformation_forward_wrapper
    def forward(self, img):
        if torch.rand(1) < self.p:
            return [TF.hflip(i) for i in img]
        return img


class RandomVerticalFlip(TT.RandomVerticalFlip):
    @transformation_forward_wrapper
    def forward(self, img):
        if torch.rand(1) < self.p:
            return [TF.vflip(i) for i in img]
        return img


class RandomRotation(TT.RandomRotation):
    @transformation_forward_wrapper
    def forward(self, imgs):
        fill = self.fill
        angle = self.get_params(self.degrees)

        out_imgs = []
        for img in imgs:
            if isinstance(img, Tensor):
                if isinstance(fill, (int, float)):
                    fill = [float(fill)] * TF.get_image_num_channels(img)
                else:
                    fill = [float(f) for f in fill]
            out_imgs.append(TF.rotate(img, angle, self.resample, self.expand, self.center, fill))
        return out_imgs


class Compose(TT.Compose):
    @transformation_forward_wrapper
    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class Resize(TT.Resize):
    @transformation_forward_wrapper
    def forward(self, img):
        return [TF.resize(i, self.size, self.interpolation, self.max_size, self.antialias) for i in img]


class RandomErasing(TT.RandomErasing):
    @transformation_forward_wrapper
    def forward(self, imgs):
        if torch.rand(1) < self.p:
            # cast self.value to script acceptable type
            if isinstance(self.value, (int, float)):
                value = [self.value]
            elif isinstance(self.value, str):
                value = None
            elif isinstance(self.value, tuple):
                value = list(self.value)
            else:
                value = self.value

            out_imgs = []
            random_state = torch.get_rng_state()
            for img in imgs:
                torch.set_rng_state(random_state)
                if value is not None and not (len(value) in (1, img.shape[-3])):
                    raise ValueError(
                        "If value is a sequence, it should have either a single value or "
                        "{} (number of input channels)".format(img.shape[-3])
                    )
                x, y, h, w, v = self.get_params(img, scale=self.scale, ratio=self.ratio, value=value)
                out_imgs.append(TF.erase(img, x, y, h, w, v, self.inplace))
            return out_imgs
        return imgs


class PILToTensor(TT.PILToTensor):
    @transformation_forward_wrapper
    def __call__(self, image):
        return [TF.pil_to_tensor(i) for i in image]


class ConvertImageDtype(TT.ConvertImageDtype):
    @transformation_forward_wrapper
    def forward(self, tensor):
        return [TF.convert_image_dtype(i, self.dtype) for i in tensor]


class Normalize(TT.Normalize):
    @transformation_forward_wrapper
    def forward(self, tensor):
        out_tensor = []
        for t in tensor:
            if t.shape[-3] == 3:
                out_tensor.append(TF.normalize(t, self.mean, self.std, self.inplace))
            elif t.shape[-3] == 1:
                out_tensor.append(TF.normalize(t, [0], [1], self.inplace))
            else:
                raise NotImplementedError
        return out_tensor


class ColorJitter(TT.ColorJitter):
    @transformation_forward_wrapper
    def forward(self, imgs):
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.get_params(
            self.brightness, self.contrast, self.saturation, self.hue
        )

        out_imgs = []
        for img in imgs:
            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    img = TF.adjust_brightness(img, brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    img = TF.adjust_contrast(img, contrast_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    img = TF.adjust_saturation(img, saturation_factor)
                elif fn_id == 3 and hue_factor is not None:
                    img = TF.adjust_hue(img, hue_factor)
            out_imgs.append(img)
        return out_imgs
