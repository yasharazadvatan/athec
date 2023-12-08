import os
from athec import misc, sharp
import glob
import json

parent_dir = "output"
sub_folder = "sharp"

if not os.path.isdir(parent_dir):
    os.mkdir(parent_dir)

if not os.path.isdir(os.path.join(parent_dir, sub_folder)):
    os.mkdir(os.path.join(parent_dir, sub_folder))

img_folder = os.path.join("image", "original")
resize_folder = os.path.join("image", "resize")
tf_folder = os.path.join("image", "transform", "sharp")

all_images = glob.glob(f"{img_folder}/*")

for img in all_images:
    image_name = img.split(os.path.sep)[-1]
    output_file = os.path.join(parent_dir, sub_folder, image_name.split('.')[0])
    if not os.path.isdir(output_file):
        os.mkdir(output_file)
    img_resized = os.path.join(resize_folder, image_name)

    '''
    Calculate sharpness as the standard deviation of Laplacian.
    save_path (optional, default None): str. If provided, a visualization will be saved to this location.
    '''
    result = sharp.attr_sharp_laplacian(img_resized,
                                        save_path=os.path.join(tf_folder, "sharp laplacian", image_name))
    misc.printd(result)
    with open(os.path.join(output_file, image_name.split('.')[0] + "-attr_sharp_laplacian.txt"), 'w') as f:
        f.write(json.dumps(result))
    '''
    Calculate sharpness with fast Fourier transform.
    save_path (optional, default None): str. If provided, a visualization will be saved to this location.
    '''
    result = sharp.attr_sharp_fft(img_resized,
                                  save_path=os.path.join(tf_folder, "sharp fft", image_name))
    misc.printd(result)
    with open(os.path.join(output_file, image_name.split('.')[0] + "-attr_sharp_fft.txt"), 'w') as f:
        f.write(json.dumps(result))
    '''
    Calculate sharpness as the standard deviation of maximum local variations of all pixels.
    save_path (optional, default None): str. If provided, a visualization will be saved to this location.
    '''
    result = sharp.attr_sharp_mlv(img_resized,
                                  save_path=os.path.join(tf_folder, "sharp mlv", image_name))
    misc.printd(result)
    with open(os.path.join(output_file, image_name.split('.')[0] + "-attr_sharp_mlv.txt"), 'w') as f:
        f.write(json.dumps(result))
    '''
    Calculate measures of depth of field based on sharpness measures after the image is partitioned into 4 × 4 blocks.
    Return:
    (1) the sharpness blur measure of the four inner blocks, divided by the average sharpness measure of all the blocks.
    (1) summary statistics of sharpness measures of all the blocks.
    (2) sharpness measure in each block, if return_raw is set to True.
    sharp_method (optional, default "laplacian"): str ("laplacian" or "fft" or "mlv") or function name. The function to calculate the blur measure for each block.
    return_summary (default False): bool. If set to True, summary statistics of sharpness measures will be returned.
    return_block (default False): bool. If set to True, the sharpness measures of all the blocks will be returned.
    '''
    result = sharp.attr_dof_block(img_resized,
                                  sharp_method=sharp.attr_sharp_fft,
                                  return_summary=True,
                                  return_block=True)
    misc.printd(result)
    with open(os.path.join(output_file, image_name.split('.')[0] + "-attr_dof_block.txt"), 'w') as f:
        f.write(json.dumps(result))
