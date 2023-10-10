import os
from athec import misc
import glob
import json

parent_dir = "output"
sub_folder = "preprocess"

if not os.path.isdir(parent_dir):
    os.mkdir(parent_dir)

if not os.path.isdir(os.path.join(parent_dir, sub_folder)):
    os.mkdir(os.path.join(parent_dir, sub_folder))

img_folder = os.path.join("image", "original")
resize_folder = os.path.join("image", "resize")

all_images = glob.glob(f"{img_folder}/*")

for img in all_images:
    image_name = img.split(os.path.sep)[-1]
    output_file = os.path.join(parent_dir, sub_folder, image_name.split('.')[0])
    if not os.path.isdir(output_file):
        os.mkdir(output_file)

    img_path = os.path.join(img_folder, image_name)
    resize_path = os.path.join(resize_folder, image_name)
    '''
    Get image mode.
    '''
    result = misc.attr_mode(img_path)
    misc.printd(result)

    with open(os.path.join(output_file, image_name.split('.')[0] + "-1.txt"), 'w') as f:
        f.write(json.dumps(result))

    '''
    Resize an image while keeping its original aspect ratio.
    Return:
    (1) the file size, width, height of the original image
    (2) the file size, width, height of the resized image.
    Parameters:
    img_path: str. The file path to the original image.
    resize_path: str. The file path where the resized image will be saved.
    max_w (optional, default -99): int. If this variable is larger than 0, the image will be resized so that its width does not exceed max_w.
    max_h (optional, default -99): int. If this variable is larger than 0, the image will be resized so that its height does not exceed max_h.
    max_side (optional, default -99): int. If this variable is larger than 0, the image will be resized so that its width and height do not exceed max_side.
    max_size (optional, default -99): int. If this variable is larger than 0, the image will be resized so its size does not exceed max_size.
    '''
    result = misc.tf_resize(img_path, resize_path, max_side=300)
    misc.printd(result)

    with open(os.path.join(output_file, image_name.split('.')[0] + "-2.txt"), 'w') as f:
        f.write(json.dumps(result))

    '''
    Calculate file size, width, height, aspect ratio, image size, image diagonal length, and file size scaled by image size.
    '''
    result = misc.attr_size(resize_path)
    misc.printd(result)

    with open(os.path.join(output_file, image_name.split('.')[0] + "-3.txt"), 'w') as f:
        f.write(json.dumps(result))







