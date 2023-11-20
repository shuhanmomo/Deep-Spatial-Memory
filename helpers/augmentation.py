import torchvision
import numbers
import random
import numpy as np
import collections
import cv2
from PIL import Image


class BrightnessJitter(object):  # 0.5 to 5 is a good range
    def __init__(self, brightness=0, consistent=True, p=0.5):
        self.brightness = self._check_input(brightness, "brightness")
        self.consistent = consistent
        self.threshold = p

    def _check_input(
        self, value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True
    ):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(
                    "If {} is a single number, it must be non negative.".format(name)
                )
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError(
                "{} should be a single number or a list/tuple with lenght 2.".format(
                    name
                )
            )

        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness):
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(
                lambda img: np.where(
                    img == 255, img, np.clip(img * brightness_factor, 0, 255)
                )
            )

        transform = torchvision.transforms.Compose(transforms)

        return transform

    def __call__(self, imgmap):
        if random.random() < self.threshold:  # do BrightnessJitter
            if self.consistent:
                transform = self.get_params(self.brightness)
                return [transform(i) for i in imgmap]
            else:
                result = []
                for img in imgmap:
                    transform = self.get_params(self.brightness)
                    result.append(transform(img))
                return result
        else:  # don't do BrightnessJitter, do nothing
            return imgmap

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        format_string += "brightness={0}".format(self.brightness)
        format_string += ")"
        return format_string


# need to test
class RandomHorizontalShift:
    def __init__(self, max_shift=30, p=0.5):
        """
        Args:
            max_shift (int): the maximum number of pixels for the horizontal shift.
            p (float): probability of applying the shift. Default is 0.5.
        """
        self.max_shift = max_shift
        self.p = p

    def __call__(self, imgmap):
        return [self.horizontal_shift(img) for img in imgmap]

    def horizontal_shift(self, img):
        """
        Shift the image horizontally by a random number of pixels and wrap around.
        Args:
            img (ndarray): the input image as a numpy array.
        Returns:
            img (ndarray): the transformed image as a numpy array.
        """
        # Check if we should apply the shift based on the probability p
        if random.random() < self.p:
            shift = random.randint(0, self.max_shift)
            shifted_np_img = np.roll(img, shift, axis=2)  # roll along width dimension
            return shifted_np_img
        return img  # return original image if not shifted


class RandomHorizontalFlip:  # choose consistent to be false
    def __init__(self, consistent=True, p=0.5):
        self.consistent = consistent
        self.threshold = p

    def __call__(self, imgmap):
        if self.consistent:
            if random.random() > self.threshold:
                return [np.flip(i, axis=[0, 2]) for i in imgmap]
            else:
                return imgmap
        else:
            result = []
            for i in imgmap:
                if random.random() > self.threshold:
                    result.append(np.flip(i, axis=[0, 2]))
                else:
                    result.append(i)
            assert len(result) == len(imgmap)
            return result


class Scale:
    def __init__(self, size, interpolation=Image.NEAREST):
        assert isinstance(size, int) or len(size) == 2
        self.size = size
        self.interpolation = interpolation

    def __call__(self, imgmap):
        num_frames, C, H, W = imgmap.shape
        resized_images = []
        for i in range(num_frames):
            img = imgmap[i, 0, :, :]  # Assuming single-channel images
            # ... resizing logic for img ...
            resized_img = cv2.resize(
                img, (self.size[1], self.size[0]), interpolation=self.interpolation
            )
            resized_img = np.expand_dims(
                resized_img, axis=0
            )  # Restore channel dimension
            resized_images.append(resized_img)

        return np.array(resized_images)


class RandomCropWithProb:
    def __init__(self, size, p=0.8, consistent=True):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = (size[0], size[1])
        self.consistent = consistent
        self.threshold = p

    def __call__(self, imgmap):
        num_frames, C, h, w = imgmap.shape
        if self.size is not None:
            th, tw = self.size
            if w == tw and h == th:
                return imgmap
            cropped_images = []
            for i in range(num_frames):
                img = imgmap[i, 0, :, :]
                if self.consistent:
                    if random.random() < self.threshold:
                        x1 = random.randint(0, w - tw)
                        y1 = random.randint(0, h - th)
                    else:
                        x1 = int(round((w - tw) / 2.0))
                        y1 = int(round((h - th) / 2.0))
                    cropped_img = img[y1 : y1 + th, x1 : x1 + tw]
                    cropped_img = np.expand_dims(
                        cropped_img, axis=0
                    )  # Restore channel dimension
                    cropped_images.append(cropped_img)
                else:
                    if random.random() < self.threshold:
                        x1 = random.randint(0, w - tw)
                        y1 = random.randint(0, h - th)
                    else:
                        x1 = int(round((w - tw) / 2.0))
                        y1 = int(round((h - th) / 2.0))
                    cropped_img = img[y1 : y1 + th, x1 : x1 + tw]
                    cropped_img = np.expand_dims(
                        cropped_img, axis=0
                    )  # Restore channel dimension
                    cropped_images.append(cropped_img)
            return np.array(cropped_images)
        else:
            return imgmap
