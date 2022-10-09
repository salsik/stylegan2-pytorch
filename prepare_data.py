import argparse
from fileinput import filename
from io import BytesIO
import multiprocessing
from functools import partial
from pkgutil import extend_path

from PIL import Image
import lmdb
from tqdm import tqdm
from torchvision import datasets
from torchvision.transforms import functional as trans_fn

import argparse
import cv2
import numpy


# by alex,,, this one resize images and save them in one output folder
def resize_and_save_tofolder(img,file, size, resample, quality=100):
    img = trans_fn.resize(img, size, resample)
    img = trans_fn.center_crop(img, size)

    # create new file  with the previous name added the resized size and same extension and file name

    #folder_name = (file.split("/")[-3]
    folder_name = '/data1/data_alex/lsun/lsun_code/kitchen256all_inn/'
    # to save before converting
     
    file_name = file.split("/")[-1]
    #extension = file_name.split(".")[-1]
    file_name = file_name.split(".")[-2] 
    
    new_file= "{}/{}_resized_{}.{}".format(folder_name,file_name,size,"png")

    #img.save(new_file)

    buffer = BytesIO()
    img.save(buffer, format="jpeg", quality=quality)
    val = buffer.getvalue()


    new_image = cv2.imdecode(
    numpy.fromstring(val, dtype=numpy.uint8), 1)
    cv2.imwrite(new_file, new_image)

    # we make this 0 when  we just want to save new images
    #val = 0

    return val


def resize_and_convert(img,file, size, resample, quality=100):
    img = trans_fn.resize(img, size, resample)
    img = trans_fn.center_crop(img, size)

    # create new file  with the previous name added the resized size and same extension and file name

    #last_folder = (file.split("/")[-3]
    
    # to save before converting 
    
    new_file= "{}_resized_{}.{}".format(file.split(".")[-2],size,file.split(".")[-1])

    #img.save(new_file)

    buffer = BytesIO()
    img.save(buffer, format="jpeg", quality=quality)
    
    val = buffer.getvalue()

    return val


def resize_multiple(
    img,file, sizes=(128, 256, 512, 1024), resample=Image.LANCZOS,quality=100
):
    imgs = []

    for size in sizes:
       
        
       # imgs.append(resize_and_convert(img,file, size, resample, quality))
        # this one is used to convert and save as png imagesm which generate an empty lmdb file
       imgs.append(resize_and_save_tofolder(img,file, size, resample, quality))

    return imgs


def resize_worker(img_file, sizes, resample):
    i, file = img_file
    img = Image.open(file)
    img = img.convert("RGB")
    out = resize_multiple(img,file=file, sizes=sizes, resample=resample)

    return i, out


def prepare(
    env, dataset, n_worker, sizes=(128, 256, 512, 1024), resample=Image.LANCZOS
):
    resize_fn = partial(resize_worker, sizes=sizes, resample=resample)

    files = sorted(dataset.imgs, key=lambda x: x[0])
    files = [(i, file) for i, (file, label) in enumerate(files)]
    total = 0

    with multiprocessing.Pool(n_worker) as pool:
        for i, imgs in tqdm(pool.imap_unordered(resize_fn, files)):
            for size, img in zip(sizes, imgs):
                key = f"{size}-{str(i).zfill(8)}".encode("utf-8")

                with env.begin(write=True) as txn:
                    f="uncomment this later"
                    txn.put(key, img)

            total += 1

        with env.begin(write=True) as txn:
            f="uncomment this later"
            txn.put("length".encode("utf-8"), str(total).encode("utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess images for model training")
    parser.add_argument("--out", type=str, help="filename of the result lmdb dataset")
    parser.add_argument(
        "--size",
        type=str,
        default="128,256,512,1024",
        help="resolutions of images for the dataset",
    )
    parser.add_argument(
        "--n_worker",
        type=int,
        default=8,
        help="number of workers for preparing dataset",
    )
    parser.add_argument(
        "--resample",
        type=str,
        default="lanczos",
        help="resampling methods for resizing images",
    )
    parser.add_argument("path", type=str, help="path to the image dataset")

    args = parser.parse_args()

    resample_map = {"lanczos": Image.LANCZOS, "bilinear": Image.BILINEAR}
    resample = resample_map[args.resample]

    sizes = [int(s.strip()) for s in args.size.split(",")]

    print(f"Make dataset of image sizes:", ", ".join(str(s) for s in sizes))

    imgset = datasets.ImageFolder(args.path)

    with lmdb.open(args.out, map_size=1024 ** 4, readahead=False) as env:
        prepare(env, imgset, args.n_worker, sizes=sizes, resample=resample)
