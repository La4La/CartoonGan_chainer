import glob
from random import randint
import numpy as np
import cv2
import chainer
import os

class PreprocessedDataset(chainer.dataset.DatasetMixin):
    def __init__(self, path, crop_size):
        self._path = glob.glob(path)
        self._crop_size = crop_size
        self._length = len(self._path)

    def __len__(self):
        return self._length

    def get_example(self, i):
        raise NotImplementedError

    @staticmethod
    def tochainer(cv2_image):
        return cv2_image.astype(np.float32).transpose((2, 0, 1))# / 255.0


class PhotoDataset(PreprocessedDataset):
    def __init__(self, path, crop_size=256):
        super().__init__(path, crop_size)

    def get_example(self, i):
        n = 1
        while True:
            try:
                n += 1
                image = cv2.imread(self._path[i], cv2.IMREAD_COLOR)
                assert image is not None
                h, w, _ = image.shape
                assert (w >= self._crop_size and h >= self._crop_size)

                # random rescale
                rescale_size = np.random.randint(self._crop_size, min(h, w) + 1)
                if h <= w:
                    h_ = int(rescale_size)
                    w_ = int(rescale_size * (w / h))
                else:
                    w_ = int(rescale_size)
                    h_ = int(rescale_size * (h / w))
                image = cv2.resize(image, (w_, h_), interpolation=cv2.INTER_AREA)

                break
            except AssertionError:
                i = (i + np.random.randint(1, self._length)) % self._length

        # random crop
        crop_w = randint(0, w_ - self._crop_size)
        crop_h = randint(0, h_ - self._crop_size)
        image = image[crop_h: crop_h + self._crop_size, crop_w: crop_w + self._crop_size, :]
        return self.tochainer(image)

    def visualizer(self, output_path='preview', n=4):
        @chainer.training.make_extension()
        def make_image(trainer):
            updater = trainer.updater
            output = os.path.join(trainer.out, output_path)
            os.makedirs(output, exist_ok=True)

            rows = []
            for i in range(n):
                photo = updater.converter(updater.get_iterator("main").next(), updater.device)

                # turn off train mode
                with chainer.using_config('train', False):
                    generated = updater.get_optimizer("gen").target(photo).data

                # convert to cv2 image
                generated = generated[0].transpose(1, 2, 0)
                photo = photo[0].transpose(1, 2, 0)

                # return image from device if necessary
                if updater.device >= 0:
                    generated = generated.get()
                    photo = photo.get()

                rows.append(np.hstack((photo, generated)).astype(np.uint8))
            cv2.imwrite(os.path.join(output, "iter_{}.png".format(updater.iteration)), np.vstack(rows))
        return make_image


class ImageDataset(PreprocessedDataset):
    def __init__(self, path, crop_size=256):
        super().__init__(path, crop_size)

    def get_example(self, i):
        while True:
            try:
                image = cv2.imread(self._path[i], cv2.IMREAD_COLOR)
                assert image is not None
                h, w, _ = image.shape
                assert (w >= self._crop_size and h >= self._crop_size)
                break
            except AssertionError:
                i = (i + np.random.randint(1, self._length)) % self._length
        crop_w = randint(0, w - self._crop_size)
        crop_h = randint(0, h - self._crop_size)
        image = image[crop_h: crop_h + self._crop_size, crop_w: crop_w + self._crop_size, :]
        smoothed = self.blur(image)
        return self.tochainer(image), self.tochainer(smoothed)

    @staticmethod
    def blur(image):
        # detect edge
        edge = cv2.Canny(image, np.random.randint(75, 125), np.random.randint(175, 225))

        # dilate edge
        d_size = np.random.randint(2, 7)
        dilated_edge = cv2.dilate(edge, np.ones((d_size, d_size), np.uint8), iterations=1)

        # gray to bgr
        dilated_edge = cv2.cvtColor(dilated_edge, cv2.COLOR_GRAY2BGR)

        # concat and blur
        _image = np.copy(image)
        b_size = np.random.randint(2, 7) * 2 + 1
        gaussian_smoothing_image = cv2.GaussianBlur(image, (b_size, b_size), 0)
        cv2.imwrite('gaussian.png', gaussian_smoothing_image)

        merged = np.where(dilated_edge > 128, gaussian_smoothing_image, _image)
        return merged
    # def blur(image):
    #     # detect edge
    #     edge = cv2.Canny(image, 100, 200)
    #     # dilate edge
    #     dilated_edge = cv2.dilate(edge, np.ones((3, 3), np.uint8), iterations=1)
    #
    #     # gray to bgr
    #     edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
    #     dilated_edge = cv2.cvtColor(dilated_edge, cv2.COLOR_GRAY2BGR)
    #
    #     # concat and blur
    #     image = image - edge + dilated_edge
    #     gaussian_smoothing_image = cv2.GaussianBlur(image, (3, 3), 0)
    #     return gaussian_smoothing_image





