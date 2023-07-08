import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2

class FacialKeypointsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, resize=True, normalize=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            resize (bool): If True, image will be resized to (224, 224) size.
            normalize (bool): If True, all the pixel value in the image will be transformed
            from [0, 255] to [0, 1]. Keypoints will be normalized using the following formula.
            keypoints = (keypoints - 100)/50
            Here, mean = 100 and standard deviation = 50
        """
        self.key_pts_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.resize = resize
        self.normalize = normalize

    def __len__(self):
        return len(self.key_pts_frame)

    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir,
                                self.key_pts_frame.iloc[idx, 0])
             
        image = cv2.imread(image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        original_shape = image.shape
        
        key_pts = self.key_pts_frame.iloc[idx, 1:].values
        key_pts = key_pts.astype('float').reshape(-1, 2)
        
        if self.resize:
            image = cv2.resize(image, (224, 224))
            key_pts = key_pts * [224/original_shape[1], 224/original_shape[0]]
        
        if self.normalize:
            image = image/255
            key_pts = (key_pts - 100)/50
        
        if(len(image.shape) == 2):
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 1)
        
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        
        sample = {'image': torch.from_numpy(image), 'keypoints': torch.from_numpy(key_pts)}

        return sample