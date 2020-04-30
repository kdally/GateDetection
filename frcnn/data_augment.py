import warnings
warnings.filterwarnings("ignore")
import cv2, copy
import numpy as np
import imgaug.augmenters as iaa


def augment(img_data, augment=True):
	""" Perform data augmentation. Horizontal fliping, Gaussian blur and contrast variations are used. """

	assert 'filepath' in img_data
	assert 'bboxes' in img_data
	assert 'width' in img_data
	assert 'height' in img_data

	img_data_aug = copy.deepcopy(img_data)

	img = cv2.imread(img_data_aug['filepath'])

	if augment:
		rows, cols = img.shape[:2]

		# apply it on 33% of the data
		if np.random.randint(0, 3) == 0:
			img = cv2.flip(img, 1)
			for bbox in img_data_aug['bboxes']:
				x1 = bbox['x1']
				x2 = bbox['x2']
				bbox['x2'] = cols - x1
				bbox['x1'] = cols - x2

		# apply it on 33% of the data
		if np.random.randint(0, 3) == 0:
			gaussian_blur = iaa.GaussianBlur(sigma=(0, 0.5))
			img = gaussian_blur.augment_image(img)

		# apply it on 33% of the data
		if np.random.randint(0, 3) == 0:
			contrast = iaa.GammaContrast(gamma=1.5)
			img = contrast.augment_image(img)

	img_data_aug['width'] = img.shape[1]
	img_data_aug['height'] = img.shape[0]

	return img_data_aug, img
