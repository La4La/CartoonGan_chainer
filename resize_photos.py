import cv2
import os, glob
import numpy as np

size = 512

root ='/mnt/sakuradata10-striped/gao/background'
photos_paths = glob.glob(os.path.join(root, 'photos', '*', '*'))
print(len(photos_paths))

out = os.path.join(root, 'photos_resized')
os.mkdir(out)

for i, p in enumerate(photos_paths):
    print(i)

    img = cv2.imread(p)
    try:
        h, w, _ = img.shape

        if h > size and w > size:
            if h >= w:
                h_ = int(size)
                w_ = int(size * (w / h))
            else:
                w_ = int(size)
                h_ = int(size * (h / w))
            _img = cv2.resize(img, (w_, h_), interpolation=cv2.INTER_AREA)
        else:
            _img = img

        new_path = os.path.join(out, os.path.basename(p))
        cv2.imwrite(new_path, _img)

    except AttributeError:
        print(p)
