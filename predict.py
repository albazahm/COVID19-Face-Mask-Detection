import warnings
import argparse
import skimage
import os
from Mask_RCNN.mrcnn.utils import Dataset, extract_bboxes
from Mask_RCNN.mrcnn.model import MaskRCNN
from Mask_RCNN.mrcnn.config import Config
from Mask_RCNN.mrcnn.visualize import display_instances
import numpy as np

warnings.filterwarnings(action='ignore')
TRAIN_SIZE=682
TEST_SIZE=171

class MaskConfig(Config):
    
    """
    Class specifying the configurations for the training of the model. Inherits Mask RCNN's Config class.
    """
    
    NAME = 'mask_cfg'
    NUM_CLASSES = 1 + 3
    IMAGES_PER_GPU = 1
    STEPS_PER_EPOCH = int(TRAIN_SIZE/IMAGES_PER_GPU)
    VALIDATION_STEPS = int(TEST_SIZE/IMAGES_PER_GPU)

class InferenceConfig(MaskConfig):
    
    """
    Class that specifies the inference configurations. Inherits the MaskConfig class.
    """
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.85

def _get_args():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--image_path",
        default='./test_images/',
        help="path to images to predict",
        type=str)

    return parser.parse_args()


def main():

    args = _get_args()
    image_path = args.image_path

    class_names = {1: 'mask', 2: 'no_mask', 3:'mask_worn_incorrectly'}
    inf_config = InferenceConfig()

    model = MaskRCNN(
        mode='inference',
        config=inf_config,
        model_dir='./'
    )

    model_path = model.find_last()
    model.load_weights(model_path, by_name=True)
        

    for filename in os.listdir(image_path):
        if filename.split('.')[-1] not in ['jpg', 'png', 'jpeg']:
            continue
        img = skimage.io.imread(image_path+filename)
        img_arr = np.array(img)
        img_arr = img_arr[:, :, :3]
        results = model.detect([img_arr], verbose=1)
        r = results[0]
        display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], figsize=(10, 10))

if __name__ == "__main__":
    
    main()
    print('Complete...')

        