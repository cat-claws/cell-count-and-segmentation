import os
import cv2
import torch
import scipy.io

import numpy as np
import torch.nn as nn

from PIL import Image

from util_hv_map import gen_targets
from utils import getEdges

def extendLabels(force = False):
	""" get horizontal/vertical maps before loading"""
	
	directories = ['CoNSeP/Train', 'CoNSeP/Test']
	setnames = ['train', 'test']
	for i in range(len(directories)):
		for index in range(len(os.listdir(os.path.join(directories[i], 'Labels')))):
			labels = scipy.io.loadmat(os.path.join(directories[i], 'Labels', setnames[i] + f'_{index + 1}.mat'))
			image = np.array(Image.open(os.path.join(directories[i], 'Images', setnames[i] + f'_{index + 1}.png')))[:,:,:3]
			editted = False
			
			if 'edge_map' not in labels:
				labels['edge_map'] = getEdges(torch.tensor(labels['inst_map'])).numpy()
				editted = True

			if ('hori_map' not in labels or 'vert_map' not in labels):
				map_dict = gen_targets(labels['inst_map'].astype(int), [1000, 1000])
				labels['hori_map'] = map_dict['h_map']
				labels['vert_map'] = map_dict['v_map']
				editted = True
			
			if editted == True or force:
				scipy.io.savemat(os.path.join(directories[i], 'Labels', setnames[i] + f'_{index + 1}.mat'), labels)

extendLabels()

class ConsepSimpleDataset(torch.utils.data.Dataset):
	def __init__(self, train = False, test = False, combine_classes = True):
		self.directory = 'CoNSeP/Train' if train else 'CoNSeP/Test'
		self.setname = 'train' if train else 'test'
		assert train + test < 2
		self.combine_classes = combine_classes

	def __len__(self):
		return len(os.listdir(os.path.join(self.directory, 'Images')))

	def __getitem__(self, index):
		# Load data and get label
		labels = scipy.io.loadmat(os.path.join(self.directory, 'Labels', self.setname + f'_{index + 1}.mat'))
		image = np.array(Image.open(os.path.join(self.directory, 'Images', self.setname + f'_{index + 1}.png')))[:,:,:3]

		image = torch.from_numpy(np.transpose(image / 255.0, (2, 0, 1))).float()
		label_inst = torch.from_numpy(labels['inst_map']).long()
		label_type = torch.from_numpy(labels['type_map']).long()
		if self.combine_classes:
			label_type.masked_fill_(label_type == 4, 3)
			label_type.masked_fill_(label_type > 4, 4)

		hori_map = torch.from_numpy(labels['hori_map']).unsqueeze(0).float()
		vert_map = torch.from_numpy(labels['vert_map']).unsqueeze(0).float()
		hv_map = torch.cat((hori_map, vert_map), dim = 0)
		
		edge_map = torch.from_numpy(labels['edge_map']).long()

		return self.transfer({'image':image, 'inst_map':label_inst, 'type_map':label_type, 'hv_map':hv_map, 'edge_map':edge_map})

	def transfer(self, data):
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		return {key:value.to(device) for key, value in data.items()}
	

class ConsepSimplePadDataset(ConsepSimpleDataset):
	def __init__(self, train = False, test = False, combine_classes = True):
		super().__init__(train, test, combine_classes)

	def __getitem__(self, index):
		# Load data and get label
		labels = scipy.io.loadmat(os.path.join(self.directory, 'Labels', self.setname + f'_{index + 1}.mat'))
		image = np.array(Image.open(os.path.join(self.directory, 'Images', self.setname + f'_{index + 1}.png')))[:,:,:3]
			
		image = torch.from_numpy(np.transpose(image / 255.0, (2, 0, 1))).float()
		label_inst = torch.from_numpy(labels['inst_map']).long()
		label_type = torch.from_numpy(labels['type_map']).long()
		if self.combine_classes:
			label_type.masked_fill_(label_type == 4, 3)
			label_type.masked_fill_(label_type > 4, 4)
		
		hori_map = torch.from_numpy(labels['hori_map']).unsqueeze(0).float()
		vert_map = torch.from_numpy(labels['vert_map']).unsqueeze(0).float()
		hv_map = torch.cat((hori_map, vert_map), dim = 0)
		
		edge_map = torch.from_numpy(labels['edge_map']).long()

		m = nn.ZeroPad2d(12)

		return self.transfer({'image':m(image), 'inst_map':m(label_inst), 'type_map':m(label_type), 'hv_map':m(hv_map), 'edge_map':m(edge_map)})

	
def simpleTransform(*target, sideLength, valid = False):
	H, W = np.array(target[0].shape[:2])
	x = np.random.randint(H * 0.7, H - sideLength) if valid else np.random.randint(H * 0.7 - sideLength)
	y = np.random.randint(W - sideLength)
	assert H * W * (H + W) > 0

	target = [t[x:x+sideLength, y:y+sideLength] for t in target]

	if np.random.random() > 0.5:
		target = [np.flip(t, 0) for t in target]
	if np.random.random() > 0.5:
		target = [np.flip(t, 1) for t in target]
	if np.random.random() > 0.5:
		target = [t.swapaxes(0, 1) for t in target]

	return tuple([t.copy() for t in target])

class ConsepSimpleTransformDataset(ConsepSimpleDataset):
	def __init__(self, train = False, test = False, valid = False, sideLength = 256, num = 200, combine_classes = True):
		super().__init__(train, test, combine_classes)
		assert train + valid + test < 2
		self.sideLength = sideLength
		self.num = num
		self.valid = valid

	def __len__(self):
		return self.num #if not test else len(os.listdir(os.path.join(self.directory, 'Images')))

	def __getitem__(self, index):
		# Load data and get label
		index = index % len(os.listdir(os.path.join(self.directory, 'Images')))
		image = np.array(Image.open(os.path.join(self.directory, 'Images', self.setname + f'_{index + 1}.png')))[:,:,:3]
		labels = scipy.io.loadmat(os.path.join(self.directory, 'Labels', self.setname + f'_{index + 1}.mat'))
		
		image, label_inst, label_type, hori_map, vert_map, edge_map = simpleTransform(image, labels['inst_map'], labels['type_map'], labels['hori_map'], labels['vert_map'], labels['edge_map'], sideLength = self.sideLength, valid = self.valid)

		label_inst = torch.from_numpy(label_inst).long()
		label_type = torch.from_numpy(label_type).long()
		image = torch.from_numpy(np.transpose(image / 255.0, (2, 0, 1))).float()
		if self.combine_classes:
			label_type.masked_fill_(label_type == 4, 3)
			label_type.masked_fill_(label_type > 4, 4)
		
		hori_map = torch.from_numpy(hori_map).unsqueeze(0).float()
		vert_map = torch.from_numpy(vert_map).unsqueeze(0).float()
		hv_map = torch.cat((hori_map, vert_map), dim = 0)
		
		edge_map = torch.from_numpy(edge_map).long()

		return self.transfer({'image':image, 'inst_map':label_inst, 'type_map':label_type, 'hv_map':hv_map, 'edge_map':edge_map})

	
def gaussian_blur(image):
	"""Apply Gaussian blur to input images."""
	ksize = np.random.randint(0, 3, size=(2,))
	ksize = tuple((ksize * 2 + 1).tolist())
	return cv2.GaussianBlur(image, ksize, sigmaX=0, sigmaY=0, borderType=cv2.BORDER_REPLICATE)

def median_blur(image):
	"""Apply median blur to input images."""
	ksize = np.random.randint(0, 3)
	ksize = ksize * 2 + 1
	return cv2.medianBlur(image, ksize)

def add_to_hue(image):
	"""Perturbe the hue of input images."""
	hue = np.random.uniform(180)
	hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	if hsv.dtype.itemsize == 1:
		# OpenCV uses 0-179 for 8-bit images
		hsv[..., 0] = (hsv[..., 0] + hue) % 180
	else:
		# OpenCV uses 0-360 for floating point images
		hsv[..., 0] = (hsv[..., 0] + 2 * hue) % 360
	return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def add_to_saturation(image):
	"""Perturbe the saturation of input images."""
	value = 1 + np.random.uniform(4)
	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	ret = image * value + (gray * (1 - value))[:, :, np.newaxis]
	return np.clip(ret, 0, 255)

def add_to_contrast(image):
	"""Perturbe the contrast of input images."""
	value = np.random.uniform()
	mean = np.mean(image, axis=(0, 1), keepdims=True)
	ret = image * value + mean * (1 - value)
	return np.clip(ret, 0, 255)

def add_to_brightness(image):
	"""Perturbe the brightness of input images."""
	value = np.random.uniform(0., 0.6)
	return np.clip(image + value, 0, 255)

class ConsepAugmentedTransformDataset(ConsepSimpleTransformDataset):
	def __init__(self, train = False, test = False, valid = False, sideLength = 256, num = 200, combine_classes = True):
		super().__init__(train, test, valid, sideLength, num, combine_classes)

	def __getitem__(self, index):
		# Load data and get label
		index = index % len(os.listdir(os.path.join(self.directory, 'Images')))
		image = np.array(Image.open(os.path.join(self.directory, 'Images', self.setname + f'_{index + 1}.png')))[:,:,:3]
		labels = scipy.io.loadmat(os.path.join(self.directory, 'Labels', self.setname + f'_{index + 1}.mat'))
		
		image, label_inst, label_type, hori_map, vert_map, edge_map = simpleTransform(image, labels['inst_map'], labels['type_map'], labels['hori_map'], labels['vert_map'], labels['edge_map'], sideLength = self.sideLength, valid = self.valid)

		label_inst = torch.from_numpy(label_inst).long()
		label_type = torch.from_numpy(label_type).long()
		image = gaussian_blur(image)
		image = median_blur(image)
		image = add_to_hue(image)
		image = add_to_saturation(image)
		image = add_to_contrast(image)
		image = add_to_brightness(image)
		image = torch.from_numpy(np.transpose(image / 255.0, (2, 0, 1))).float()
		if self.combine_classes:
			label_type.masked_fill_(label_type == 4, 3)
			label_type.masked_fill_(label_type > 4, 4)
		
		hori_map = torch.from_numpy(hori_map).unsqueeze(0).float()
		vert_map = torch.from_numpy(vert_map).unsqueeze(0).float()
		hv_map = torch.cat((hori_map, vert_map), dim = 0)
		
		edge_map = torch.from_numpy(edge_map).long()

		return self.transfer({'image':image, 'inst_map':label_inst, 'type_map':label_type, 'hv_map':hv_map, 'edge_map':edge_map})
