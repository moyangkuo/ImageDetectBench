from __future__ import absolute_import, division, print_function, unicode_literals

import logging, sys, io, torch, cv2
from torchvision import transforms
from PIL import Image
from typing import Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
from tqdm.auto import tqdm

from art.config import ART_NUMPY_DTYPE
from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator
from art.estimators.classification import ClassifierMixin
from art.utils import to_categorical, check_and_transform_label_format, get_labels_np_array
from src.image_transform import FROST_DIR

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE

logging.basicConfig(format='%(asctime)s (%(levelname)s) %(message)s', level=logging.INFO,
                    datefmt='%d.%m.%Y %H:%M:%S', 
                    handlers=[logging.StreamHandler(), 
                              logging.FileHandler("./log.txt")])
logger = logging.getLogger()
NO_PERTURB_REQUIRED = "No perturbation required"


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


def plasma_fractal(mapsize=256, wibbledecay=3):
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


class HopSkipJump(EvasionAttack):

    attack_params = EvasionAttack.attack_params + [
        "targeted",
        "norm",
        "max_iter",
        "max_eval",
        "init_eval",
        "init_size",
        "curr_iter",
        "batch_size",
        "verbose",
    ]
    _estimator_requirements = (BaseEstimator, ClassifierMixin)

    def __init__(
        self,
        classifier,
        batch_size: int = 64,
        targeted: bool = False,
        norm: Union[int, float, str] = 2,
        max_iter: int = 50,
        max_eval: int = 10000,
        init_eval: int = 100,
        init_size: int = 100,
        verbose: bool = True,
        use_init_img: bool = None,
    ) -> None:
        super().__init__(estimator=classifier)
        self._targeted = targeted
        self.norm = norm
        self.max_iter = max_iter
        self.max_eval = max_eval
        self.init_eval = init_eval
        self.init_size = init_size
        self.curr_iter = 0
        self.batch_size = batch_size
        self.verbose = verbose
        self._check_params()
        self.curr_iter = 0
        self.estimator._input_shape = [64, 64]
        self.use_init_img = use_init_img
        assert use_init_img is not None
        if norm == 2:
            self.theta = 0.01 / np.sqrt(np.prod(self.estimator.input_shape))
        else:
            self.theta = 0.01 / np.prod(self.estimator.input_shape)
        
        self.inverse_normalize = transforms.Compose([])
        self.normalize = transforms.Compose([])

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        mask = kwargs.get("mask")

        if y is None:
            if self.targeted:
                raise ValueError("Target labels `y` need to be provided for a targeted attack.")

            y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size, **kwargs))

        y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes)
        
        resume = kwargs.get("resume")

        if resume is not None and resume:
            start = self.curr_iter
        else:
            start = 0

        if mask is not None:
            if len(mask.shape) == len(x.shape):
                mask = mask.astype(ART_NUMPY_DTYPE)
            else:
                mask = np.array([mask.astype(ART_NUMPY_DTYPE)] * x.shape[0])
        else:
            mask = np.array([None] * x.shape[0])

        if self.estimator.clip_values is not None:
            clip_min, clip_max = self.estimator.clip_values
        else:
            clip_min, clip_max = np.min(x), np.max(x)

        preds = np.argmax(self.estimator.predict(x, batch_size=self.batch_size, **kwargs), axis=1)

        x_adv_init = kwargs.get("x_adv_init")
        
        if x_adv_init is not None:
            print(x_adv_init.shape)
            for i in range(x.shape[0]):
                if mask[i] is not None:
                    x_adv_init[i] = x_adv_init[i] * mask[i] + x[i] * (1 - mask[i])

            init_preds = np.argmax(self.estimator.predict(x_adv_init, batch_size=self.batch_size, **kwargs), axis=1)
            print("init_preds:", init_preds)

        else:
            init_preds = [None] * len(x)
            x_adv_init = [None] * len(x)

        if self.targeted and y is None:
            raise ValueError("Target labels `y` need to be provided for a targeted attack.")

        x_adv = x.astype(ART_NUMPY_DTYPE)

        y = np.argmax(y, axis=1)

        num_queries_ls = kwargs.get("num_queries_ls")
        if num_queries_ls is None:
            num_queries_ls = np.zeros((len(x_adv)))
            print("Initialize the list of the number of queries.")
            print(num_queries_ls)
            print("")

        for ind, val in enumerate(tqdm(x_adv, desc="HopSkipJump", disable=not self.verbose)):
            self.curr_iter = start

            if self.estimator.clip_values is None:
                clip_min = np.min(val)
                clip_max = np.max(val)   

            if self.targeted:
                x_adv[ind], num_queries = self._perturb(
                    x=val,
                    y=y[ind],
                    y_p=preds[ind],
                    init_pred=init_preds[ind],
                    adv_init=x_adv_init[ind],
                    mask=mask[ind],
                    clip_min=clip_min,
                    clip_max=clip_max,
                    **kwargs,
                )

            else:
                x_adv[ind], num_queries = self._perturb(
                    x=val,
                    y=-1,
                    y_p=preds[ind],
                    init_pred=init_preds[ind],
                    adv_init=x_adv_init[ind],
                    mask=mask[ind],
                    clip_min=clip_min,
                    clip_max=clip_max,
                    **kwargs,
                )
            num_queries_ls[ind] += num_queries

        return x_adv, num_queries_ls

    def _perturb(
        self,
        x: np.ndarray, 
        y: int,
        y_p: int,
        init_pred: int,
        adv_init: np.ndarray,
        mask: Optional[np.ndarray],
        clip_min: float,
        clip_max: float, 
        **kwargs,
    ) -> np.ndarray:
        initial_sample, init_num_queries = self._init_sample(x, y, y_p, init_pred, adv_init, mask, clip_min, clip_max, **kwargs)

        if initial_sample is None:
            return x, -1
        elif initial_sample[1] == NO_PERTURB_REQUIRED:
            return x, 0

        x_adv, attack_num_queries = self._attack(initial_sample[0], x, initial_sample[1], mask, clip_min, clip_max, **kwargs)

        return x_adv, attack_num_queries+init_num_queries

    def _init_sample(
        self,
        x: np.ndarray,
        y: int,
        y_p: int,
        init_pred: int,
        adv_init: np.ndarray,
        mask: Optional[np.ndarray],
        clip_min: float,
        clip_max: float,
        **kwargs, 
    ):
        initial_sample = None

        init_num_queries = 0
        frost_blur_params = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
        noise_params = [0.0,0.005,0.01,0.02,0.04,0.06,0.08,0.1,0.15,0.3,0.5,0.8,1.2]
        if self.targeted:
            if y == y_p:
                print("y == y_p: return (x, NO_PERTURB_REQUIRED), 0")
                return (x, NO_PERTURB_REQUIRED), 0

            print("adv_init is not None and init_pred == y:", adv_init is not None, init_pred == y)
            print("init_pred:", init_pred, "y:", y)
            if adv_init is not None and init_pred == y:
                print("adv_init is not None and init_pred == y: return (adv_init.astype(ART_NUMPY_DTYPE), init_pred), 0")
                return (adv_init.astype(ART_NUMPY_DTYPE), init_pred), 0
            if self.use_init_img == True:
                if y == 1:
                    init_img_path = "../fake_real_img_dataset/stable_diffusion/val/val_class/2.png"
                if y == 0:
                    init_img_path = "../fake_real_img_dataset/mscoco/val/val_class/000000001580.png"
                tf = transforms.Compose([transforms.CenterCrop((224, 224)),
                            transforms.CenterCrop(64),
                            transforms.ToTensor(), 
                            transforms.Grayscale()])
                init_img = tf(Image.open(init_img_path)).numpy()
                random_class = np.argmax(
                            self.estimator.predict(np.array([init_img]), batch_size=self.batch_size, **kwargs),
                            axis=1,
                        )[0] 
                init_num_queries += 1
                if random_class == y:
                    random_img, num_queries_bs = self._binary_search(
                        current_sample=init_img, original_sample=x,
                        target=y, norm=2, clip_min=clip_min, clip_max=clip_max, threshold=0.001, **kwargs)
                    initial_sample = (random_img, random_class)
                    init_num_queries += num_queries_bs
                    logger.info(f"Found initial adversarial image for untargeted attack WITHOUT perturbation.")
                    return initial_sample, init_num_queries
            
            for init_size_iter in range(len(frost_blur_params)):
                c = float(frost_blur_params[init_size_iter])
                for variance in noise_params:
                    break_condition = c <= 0.5
                    if break_condition:
                        variance *= 0
                    random_np = torch.randn_like(torch.tensor(x)).numpy() * variance
                    
                    frost_blur_img = self._frost_blur(x, c=c, clip_min=clip_min, clip_max=clip_max)
                    random_img = np.clip(frost_blur_img+random_np, clip_min, clip_max)
                    
                    if mask is not None:
                        random_img = random_img * mask + x * (1 - mask)
                    raw_logits = self.estimator.predict(np.array([random_img]), batch_size=self.batch_size, **kwargs)
                    random_class = np.argmax(
                        raw_logits,
                        axis=1,
                    )[0]
                    init_num_queries += 1

                    if random_class == y:
                        random_img, num_queries_bs = self._binary_search(
                            current_sample=random_img,
                            original_sample=x,
                            target=y,
                            norm=2,
                            clip_min=clip_min,
                            clip_max=clip_max,
                            threshold=0.001, **kwargs,
                        )
                        initial_sample = (random_img, random_class)
                        init_num_queries += num_queries_bs
                        logger.info(f"Found initial adversarial image for untargeted attack, frost_blur={c}, variance={variance}.")
                        return initial_sample, init_num_queries
                    if break_condition:
                        break
            else:
                logger.warning("Failed to draw a random image that is adversarial, attack failed.")

        else:
            if adv_init is not None and init_pred != y_p:
                return (adv_init.astype(ART_NUMPY_DTYPE), y_p), 0

            for _ in range(self.init_size):
                random_img = self._jpeg_compression(x, c=c, clip_min=clip_min, clip_max=clip_max)
                if mask is not None:
                    random_img = random_img * mask + x * (1 - mask)

                random_class = np.argmax(
                    self.estimator.predict(np.array([random_img]), batch_size=self.batch_size, **kwargs),
                    axis=1,
                )[0]
                init_num_queries += 1
                if random_class != y_p:
                    random_img, num_queries_bs = self._binary_search(
                        current_sample=random_img,
                        original_sample=x,
                        target=y_p,
                        norm=2,
                        clip_min=clip_min,
                        clip_max=clip_max,
                        threshold=0.001, **kwargs,
                    )
                    initial_sample = (random_img, y_p)
                    init_num_queries += num_queries_bs

                    logger.info(f"Found initial adversarial image for untargeted attack, jpeg_compression={c}.")
                    break
            else:
                logger.warning("Failed to draw a random image that is adversarial, attack failed.")

        return initial_sample, init_num_queries

    def _attack(
        self,
        initial_sample: np.ndarray, 
        original_sample: np.ndarray,
        target: int,
        mask: Optional[np.ndarray],
        clip_min: float,
        clip_max: float, **kwargs,
    ) -> np.ndarray:
        current_sample = initial_sample

        total_num_queries_bs = 0
        total_num_queries_est = 0 
        total_num_queries_lr_decay = 0

        for _ in range(self.max_iter):
            delta = self._compute_delta(
                current_sample=current_sample,
                original_sample=original_sample,
                clip_min=clip_min,
                clip_max=clip_max,
            )

            current_sample, num_queries_bs = self._binary_search(
                current_sample=current_sample, 
                original_sample=original_sample,
                norm=self.norm,
                target=target,
                clip_min=clip_min,
                clip_max=clip_max, **kwargs,
            )
            total_num_queries_bs += num_queries_bs

            num_queries_est = min(int(self.init_eval * np.sqrt(self.curr_iter + 1)), self.max_eval)
            update = self._compute_update(
                current_sample=current_sample, 
                num_eval=num_queries_est,
                delta=delta,
                target=target,
                mask=mask,
                clip_min=clip_min,
                clip_max=clip_max, **kwargs,
            )
            total_num_queries_est += num_queries_est
            if self.norm == 2:
                dist = np.linalg.norm(original_sample - current_sample)
            else:
                dist = np.max(abs(original_sample - current_sample))

            epsilon = 2.0 * dist / np.sqrt(self.curr_iter + 1)
            success = False

            while not success:
                epsilon /= 2.0
                potential_sample = current_sample + epsilon * update
                success = self._adversarial_satisfactory(
                    samples=potential_sample[None], 
                    target=target,
                    clip_min=clip_min,
                    clip_max=clip_max, **kwargs,
                )
                total_num_queries_lr_decay += 1

            current_sample = np.clip(potential_sample, clip_min, clip_max)

            self.curr_iter += 1

            if np.isnan(current_sample).any(): 
                logger.warning("NaN detected in sample, returning original sample.")
                total_num_queries = total_num_queries_bs+total_num_queries_est+total_num_queries_lr_decay
                return original_sample, total_num_queries

        total_num_queries = total_num_queries_bs+total_num_queries_est+total_num_queries_lr_decay
        return current_sample, total_num_queries

    def _binary_search(
        self,
        current_sample: np.ndarray, 
        original_sample: np.ndarray,
        target: int,
        norm: Union[int, float, str],
        clip_min: float,
        clip_max: float,
        threshold: Optional[float] = None, **kwargs,
    ) -> np.ndarray:
        if norm == 2:
            (upper_bound, lower_bound) = (1, 0)

            if threshold is None:
                threshold = self.theta

        else:
            (upper_bound, lower_bound) = (
                np.max(abs(original_sample - current_sample)),
                0,
            )

            if threshold is None:
                threshold = np.minimum(upper_bound * self.theta, self.theta)

        num_queries_bs = 0
        while (upper_bound - lower_bound) > threshold:
            alpha = (upper_bound + lower_bound) / 2.0
            interpolated_sample = self._interpolate(
                current_sample=current_sample,
                original_sample=original_sample,
                alpha=alpha,
                norm=norm,
            )

            satisfied = self._adversarial_satisfactory(
                samples=interpolated_sample[None],
                target=target,
                clip_min=clip_min,
                clip_max=clip_max, **kwargs,
            )[0]
            lower_bound = np.where(satisfied == 0, alpha, lower_bound)
            upper_bound = np.where(satisfied == 1, alpha, upper_bound)

            num_queries_bs += 1

        result = self._interpolate(
            current_sample=current_sample,
            original_sample=original_sample,
            alpha=upper_bound,
            norm=norm,
        )

        return result, num_queries_bs

    def _compute_delta(
        self,
        current_sample: np.ndarray,
        original_sample: np.ndarray,
        clip_min: float,
        clip_max: float,
    ) -> float:
        if self.curr_iter == 0:
            return 0.1 * (clip_max - clip_min)

        if self.norm == 2:
            dist = np.linalg.norm(original_sample - current_sample)
            delta = np.sqrt(np.prod(self.estimator.input_shape)) * self.theta * dist
        else:
            dist = np.max(abs(original_sample - current_sample))
            delta = np.prod(self.estimator.input_shape) * self.theta * dist

        return delta

    def _compute_update(
        self,
        current_sample: np.ndarray,
        num_eval: int,
        delta: float,
        target: int,
        mask: Optional[np.ndarray],
        clip_min: float,
        clip_max: float, **kwargs,
    ) -> np.ndarray:
        rnd_noise_shape = [num_eval] + list(self.estimator.input_shape)
        if self.norm == 2:
            rnd_noise = np.random.randn(*rnd_noise_shape).astype(ART_NUMPY_DTYPE)
        else:
            rnd_noise = np.random.uniform(low=-1, high=1, size=rnd_noise_shape).astype(ART_NUMPY_DTYPE)

        if mask is not None:
            rnd_noise = rnd_noise * mask

        rnd_noise = rnd_noise / np.sqrt(
            np.sum(
                rnd_noise ** 2,
                axis=tuple(range(len(rnd_noise_shape)))[1:],
                keepdims=True,
            )
        )
        eval_samples = np.clip(current_sample + delta * rnd_noise, clip_min, clip_max)
        rnd_noise = (eval_samples - current_sample) / delta

        satisfied = self._adversarial_satisfactory(
            samples=eval_samples, target=target, clip_min=clip_min, clip_max=clip_max, **kwargs,
        )
        f_val = 2 * satisfied.reshape([num_eval] + [1] * len(self.estimator.input_shape)) - 1.0
        f_val = f_val.astype(ART_NUMPY_DTYPE)

        if np.mean(f_val) == 1.0:
            grad = np.mean(rnd_noise, axis=0)
        elif np.mean(f_val) == -1.0:
            grad = -np.mean(rnd_noise, axis=0)
        else:
            f_val -= np.mean(f_val)
            grad = np.mean(f_val * rnd_noise, axis=0)

        if self.norm == 2:
            result = grad / np.linalg.norm(grad)
        else:
            result = np.sign(grad)

        return result

    def _adversarial_satisfactory(
        self, samples: np.ndarray, target: int, clip_min: float, clip_max: float, **kwargs
    ) -> np.ndarray:
        samples = np.clip(samples, clip_min, clip_max)
        preds = np.zeros((len(samples)))

        try:
            st = 0
            while st<len(samples):
                cur_batch_size = min(self.batch_size, len(samples)-st)
                preds[st:st+cur_batch_size] = np.argmax(self.estimator.predict(samples[st:st+cur_batch_size], batch_size=self.batch_size, **kwargs), axis=1)
                st += cur_batch_size
        except Exception as e:
            print("Exception occurs.", repr(e))
            raise e

        if self.targeted:
            result = preds == target
        else:
            result = preds != target

        return result

    @staticmethod
    def _interpolate(
        current_sample: np.ndarray, original_sample: np.ndarray, alpha: float, norm: Union[int, float, str]
    ) -> np.ndarray:

        if norm == 2:
            result = (1 - alpha) * original_sample + alpha * current_sample
        else:
            result = np.clip(current_sample, original_sample - alpha, original_sample + alpha)

        return result

    def _check_params(self) -> None:
        if self.norm not in [2, np.inf, "inf"]:
            raise ValueError('Norm order must be either 2, `np.inf` or "inf".')

        if not isinstance(self.max_iter, int) or self.max_iter < 0:
            raise ValueError("The number of iterations must be a non-negative integer.")

        if not isinstance(self.max_eval, int) or self.max_eval <= 0:
            raise ValueError("The maximum number of evaluations must be a positive integer.")

        if not isinstance(self.init_eval, int) or self.init_eval <= 0:
            raise ValueError("The initial number of evaluations must be a positive integer.")

        if self.init_eval > self.max_eval:
            raise ValueError("The maximum number of evaluations must be larger than the initial number of evaluations.")

        if not isinstance(self.init_size, int) or self.init_size <= 0:
            raise ValueError("The number of initial trials must be a positive integer.")

        if not isinstance(self.verbose, bool):
            raise ValueError("The argument `verbose` has to be of type bool.")
    
    def _jpeg_compression(self, x, c, clip_min, clip_max):
        output = io.BytesIO()
        self.inverse_normalize(torch.tensor(x).unsqueeze(0)).save(output, 'JPEG', quality=int(c))
        x = Image.open(output)
        return np.clip(self.normalize(x).squeeze().numpy(), clip_min, clip_max)

    def _elastic_blur(self, x, c, clip_min, clip_max):
        elastic_blur = transforms.ElasticTransform(alpha=abs(c))
        return np.clip(np.array((elastic_blur(torch.tensor(x).unsqueeze(0)*0.5+0.5)-0.5).squeeze()/0.5), clip_min, clip_max)
    
    def _frost_blur(self, x, c, clip_min, clip_max):
        idx = np.random.randint(0, 5)
        filename = [os.path.join(FROST_DIR, 'frost/frost1.png'),
                    os.path.join(FROST_DIR, 'frost/frost2.png'),
                    os.path.join(FROST_DIR, 'frost/frost3.png'),
                    os.path.join(FROST_DIR, 'frost/frost4.jpg'),
                    os.path.join(FROST_DIR, 'frost/frost5.jpg'),
                    os.path.join(FROST_DIR, 'frost/frost6.jpg')][idx]
        frost_img = cv2.imread(filename)
        frost_img = cv2.resize(frost_img, (int(1.5 * np.array(x).shape[0]), int(1.5 * np.array(x).shape[1]))).astype(np.float32) / 255
        x_start, y_start = np.random.randint(0, frost_img.shape[0] - np.array(x).shape[0]), np.random.randint(0, frost_img.shape[1] - np.array(x).shape[1])
        frost_img = frost_img[x_start:x_start + np.array(x).shape[0], y_start:y_start + np.array(x).shape[1]][:, :, 0]
        ret_val = np.clip((x*0.5+0.5 + (c * frost_img) - 0.5)/0.5, clip_min, clip_max)
        return ret_val
    
    def _defocus_blur(self, x, c, clip_min, clip_max):
        x = x*0.5 + 0.5
        kernel = disk(radius=c, alias_blur=0.5)

        channels = []
        for d in range(1):
            channels.append(cv2.filter2D(x[:, :], -1, kernel))
        channels = (np.array(channels) - 0.5) / 0.5
        return np.clip(channels, clip_min, clip_max)


    def _fog(self, x, c, clip_min, clip_max):
        x = x*0.5 + 0.5
        
        orig_shape = np.max(x.shape)
        max_val = x.max()
        pert = plasma_fractal(mapsize=x.shape[1], wibbledecay=2.0)[:x.shape[1], :x.shape[1]]

        x += c * pert

        return np.clip(np.array((x * max_val / (max_val + c) - 0.5)/0.5), clip_min, clip_max)


import argparse
import pickle
from pathlib import Path
import os, json
import datetime
import time, lpips
import concurrent.futures

from torchvision import transforms
import numpy as np
import pandas as pd
from cytoolz import functoolz
from torchmetrics import StructuralSimilarityIndexMeasure

from src.data import array_from_imgdir
from src.image import dct, log_scale

from art.estimators.classification.scikitlearn import ScikitlearnLogisticRegression

from utils.env_vars import PAIR_ABBREV


loss_fn_alex = lpips.LPIPS(net='alex')
loss_fn_vgg = lpips.LPIPS(net='vgg')


class MyScikitlearnLogisticRegression(ScikitlearnLogisticRegression):
    def __init__(
        self,
        model, scaler,
        clip_values = None,
        preprocessing_defences = None,
        postprocessing_defences = None,
        preprocessing = (0.0, 1.0),
    ) -> None:
        super(MyScikitlearnLogisticRegression, self).__init__(
            model=model, clip_values=clip_values, preprocessing_defences=preprocessing_defences, 
            postprocessing_defences=postprocessing_defences, preprocessing=preprocessing)
        self.scaler = scaler
    
    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)
        x_preprocessed = tf_func(x_preprocessed)
        x_preprocessed = x_preprocessed.reshape((len(x_preprocessed), -1))
        x_preprocessed = self.scaler.transform(x_preprocessed)

        if self._use_logits:
            if callable(getattr(self.model, "predict_log_proba", None)):
                y_pred = self.model.predict_log_proba(x_preprocessed)
            else: 
                raise ValueError(
                    "Argument `use_logits` was True but classifier model does not have callable" "`predict_log_proba`."
                )
        elif callable(getattr(self.model, "predict_proba", None)):
            y_pred = self.model.predict_proba(x_preprocessed)
        elif callable(getattr(self.model, "predict", None)):
            y_pred = to_categorical(
                self.model.predict(x_preprocessed),
                nb_classes=self._get_nb_classes(),
            )
        else:
            raise ValueError("The provided model does not have methods `predict_proba` or `predict`.")

        predictions = self._apply_postprocessing(preds=y_pred, fit=False)

        return predictions

PD_COLUMNS = ['image_name', 'query', 'l2', 'linf', 'SSIM', 'lpips_alex', 'lpips_vgg']
REAL_DIR_NAMES = {"flickr10k", "mscoco", "googlecc_dalle3/googlecc", "textcap_gpt_11k_human"}
FAKE_DIR_NAMES = {"deepfloyd_IF_flickr30k", "stable_diffusion", "googlecc_dalle3/dalle3", "textcap_gpt_11k_synthesis_gpt4cap"}

INT_TO_CLASS = {1: "fake", 0: "real"}

tf_name, tf_func = "log(DCT)", functoolz.compose_left(dct, log_scale)


def thread_get_x_adv(output_dir, use_init_img, query_budget, xxfadg, norm,
                     model_pytorch_cls, img_numpy, label_list, path_list,
                     ssim, inverse_norm, start_time):
    local_results = pd.DataFrame(columns=PD_COLUMNS)
    blackbox_attack = HopSkipJump(use_init_img=use_init_img,
        max_iter=20, init_eval=5, max_eval=1000, 
        classifier=model_pytorch_cls, batch_size=1, targeted=True, 
        norm=norm, init_size=10)
    num_queries_ls = np.zeros(1)
    while_iter = 0
    while num_queries_ls <= query_budget and num_queries_ls >= 0:
        x_adv, num_queries_ls = blackbox_attack.generate(
            x=np.array([img_numpy[xxfadg]]), 
            x_adv_init=None if num_queries_ls == 0 else x_adv,
            num_queries_ls=num_queries_ls, 
            resume=True,
            y=np.array([1-label_list[xxfadg]]), 
            num_workers=0)
        curr_logits = model_pytorch_cls.predict(x_adv, batch_size=1)
        print("curr_logits", curr_logits.flatten())
        adv_cls = np.argmax(curr_logits, axis=-1)
        print("clean_label:", label_list[xxfadg], "adv_cls:", adv_cls, "num_queries_ls:", num_queries_ls)
        ssim_val = ssim(inverse_norm(torch.tensor(x_adv).unsqueeze(0)),
            inverse_norm(torch.tensor(img_numpy[xxfadg]).unsqueeze(0).unsqueeze(0)))
        ssim_val = float(
            ssim_val.cpu().item()
        )
        loss_alex = float(loss_fn_alex((torch.tensor(x_adv)-0.5)/0.5, (torch.tensor(img_numpy[xxfadg]).unsqueeze(0)-0.5)/0.5))
        loss_vgg = float(loss_fn_vgg((torch.tensor(x_adv)-0.5)/0.5, (torch.tensor(img_numpy[xxfadg]).unsqueeze(0)-0.5)/0.5))
        error_l2 = np.linalg.norm(x_adv.flatten() - img_numpy[xxfadg].flatten())
        error_linf = np.max(abs(x_adv.flatten() - img_numpy[xxfadg].flatten()))
        local_results.loc[len(local_results)] = ["/".join(path_list[xxfadg].split("/")[-4:]), int(num_queries_ls), error_l2, error_linf, ssim_val, loss_alex, loss_vgg]
        print(f"Image {xxfadg}, while_iter {while_iter}, duration {time.time() - start_time}, curr_time {datetime.datetime.now()}")
        while_iter += 1
        with open(os.path.join(output_dir, f'imgs/img_{xxfadg}_num_queries_{int(num_queries_ls)}.pkl'), 'wb') as file: 
            pickle.dump({"x_adv_single": x_adv, "path": path_list[xxfadg], 
                            "orig_label": label_list[xxfadg]}, 
                        file)
        if num_queries_ls == 0:
            break
    return {"results_df": local_results, "x_adv": x_adv}


def get_abbreviation(train_dirs, test_dirs):
    train_str = ""
    if train_dirs is not None:
        for train_dir in train_dirs:
            train_str = train_str + PAIR_ABBREV[train_dir] + "_"
        train_str = train_str[0:-1]
    test_str = ""
    if test_dirs is not None:
        for test_dir in test_dirs:
            test_str = test_str + PAIR_ABBREV[test_dir] + "_"
        test_str = test_str[0:-1]
    return train_str, test_str


def main(args):
    if args.norm == "2": args.norm = int(args.norm)

    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device='cpu')

    start = time.time()
    temp = REAL_DIR_NAMES.union(FAKE_DIR_NAMES)
    for img_dir_test in args.test_dirs:
        assert img_dir_test in temp
    output_dir = args.output_root
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "imgs").mkdir(parents=True, exist_ok=True)
    with open(os.path.join(str(output_dir), 'commandline_args.txt'), 'w') as f:
        args.image_root = str(args.image_root)
        args.output_root = str(args.output_root)
        json.dump(args.__dict__, f, indent=2)
        args.image_root = Path(args.image_root)
        args.output_root = Path(args.output_root)

    transforms_list_base = [transforms.CenterCrop((224, 224)),
                            transforms.CenterCrop((args.crop_size, args.crop_size)),
                            transforms.ToTensor(), 
                            transforms.Grayscale()]
    transforms_list_test = transforms_list_base

    inverse_norm = transforms.Compose([])
    
    cache_file = Path(args.cache_file_path)

    print(f"loading already trained model, cache_file={cache_file}")
    with open(cache_file, "rb") as f:
        best_model = pickle.load(f)
    regression_model = MyScikitlearnLogisticRegression(best_model[1], scaler=best_model[3], clip_values=(-1, 1))
    
    num_samples_per_dataset = min(int(args.num_samples_total / len(args.test_dirs)), args.num_test)
    num_samples_per_label = int(num_samples_per_dataset)
    ret_list = []
    path_to_csv = os.path.join(output_dir, 'results.csv')

    for test_img_dir in args.test_dirs:
        assert cache_file.exists() == True
        ret = array_from_imgdir(
            args.image_root / test_img_dir / "val" / "val_class",
            num_samples=int(args.num_test/len(args.test_dirs)),
            num_workers=args.num_workers, 
            transforms_list=transforms_list_test
        )
        if test_img_dir in REAL_DIR_NAMES:
            ret["y_test"] = np.array([0] * len(ret["imgs"]))
        elif test_img_dir in FAKE_DIR_NAMES:
            ret["y_test"] = np.array([1] * len(ret["imgs"]))
        ret_list.append(ret)

    y_test = np.concatenate((ret_list[0]["y_test"], ret_list[1]["y_test"]))
    assert len(args.test_dirs) == 2 and 1 in set(y_test) and 0 in set(y_test)
    paths = ret_list[0]["paths"] + ret_list[1]["paths"]

    x_test = np.concatenate(
        (ret_list[0]["imgs"][0:num_samples_per_label], 
        ret_list[1]["imgs"][0:num_samples_per_label])
        )
    y_test = np.concatenate(
        (ret_list[0]["y_test"][0:num_samples_per_label], 
        ret_list[1]["y_test"][0:num_samples_per_label])
        )
    paths = ret_list[0]["paths"][0:num_samples_per_label] + ret_list[1]["paths"][0:num_samples_per_label]
    print(len(x_test))
    print(f"Finished data loading, loaded test data {','.join(args.test_dirs)}, time since start of main {time.time() - start}")
    
    label_list = np.array(y_test)
    
    results = pd.DataFrame(columns=PD_COLUMNS)
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_threads) as executor:
        start_from = args.start_from_idx
        future_to_idx = {executor.submit(
            thread_get_x_adv, output_dir, bool(args.use_init_img), args.query_budget, xxfadg, args.norm,
            regression_model, x_test, label_list, paths,
            ssim, inverse_norm, start): (paths[xxfadg], xxfadg, label_list[xxfadg]) 
            for xxfadg in range(start_from, x_test.shape[0])}
        for future in concurrent.futures.as_completed(future_to_idx):
            local_path, local_idx, local_label = future_to_idx[future]
            ret = future.result()
            local_results, local_x_adv = ret["results_df"], ret["x_adv"]
            results = pd.concat([results, local_results], ignore_index=True)
            print(f"Got results from idx={local_idx}, len(results)={len(results)}")
            results.to_csv(path_to_csv, index=False)
    return
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", type=Path, default="path/to/fake_real_img_dataset",
                        help="Root of image directory containing 'train', 'val', and test.")
    parser.add_argument('--use_init_img', type=bool, default=False, help="use init adv img?", action=argparse.BooleanOptionalAction)
    parser.add_argument("--output_root", type=Path, default="output_dir", help="Output directory.")
    parser.add_argument("--test_dirs", nargs="+", required=True, help="Names of directories in 'train' and 'val'.")
    parser.add_argument("--crop_size", type=int, default=64, help="Size the image will be cropped to.")
    parser.add_argument("--cache_file_path", type=str, required=True, help="Location of load model checkpoint.")
    parser.add_argument("--num_samples_total", type=int, default=100, help="Total number of samples")
    parser.add_argument("--num_test", type=int, default=2000)
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers (default: 8).")

    parser.add_argument('--norm', default="inf", choices=["inf", "2"])
    parser.add_argument('--query_budget', default=10_000, type=int)
    parser.add_argument('--max_threads', default=1, type=int, choices=[1], help='No. of threads in threadpool. Does not support multithreading')
    parser.add_argument('--start_from_idx', default=0, type=int, help='Which image to start from.')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)