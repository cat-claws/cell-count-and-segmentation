"""Wrapped by Tianqi Xiao"""

import math

import cv2
import matplotlib.cm as cm
import numpy as np

from scipy import ndimage
from scipy.ndimage import measurements
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import affine_transform, map_coordinates

from skimage import morphology as morph

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

def get_bounding_box(img):
	"""Get bounding box coordinate information."""
	rows = np.any(img, axis=1)
	cols = np.any(img, axis=0)
	rmin, rmax = np.where(rows)[0][[0, -1]]
	cmin, cmax = np.where(cols)[0][[0, -1]]
	# due to python indexing, need to add 1 to max
	# else accessing will be 1px in the box, not out
	rmax += 1
	cmax += 1
	return [rmin, rmax, cmin, cmax]

def cropping_center(x, crop_shape, batch=False):
	"""Crop an input image at the centre.
	Args:
		x: input array
		crop_shape: dimensions of cropped array

	Returns:
		x: cropped array

	"""
	orig_shape = x.shape
	if not batch:
		h0 = int((orig_shape[0] - crop_shape[0]) * 0.5)
		w0 = int((orig_shape[1] - crop_shape[1]) * 0.5)
		x = x[h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]
	else:
		h0 = int((orig_shape[1] - crop_shape[0]) * 0.5)
		w0 = int((orig_shape[2] - crop_shape[1]) * 0.5)
		x = x[:, h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]
	return x

def center_pad_to_shape(img, size, cval=255):
	"""Pad input image."""
	# rounding down, add 1
	pad_h = size[0] - img.shape[0]
	pad_w = size[1] - img.shape[1]
	pad_h = (pad_h // 2, pad_h - pad_h // 2)
	pad_w = (pad_w // 2, pad_w - pad_w // 2)
	if len(img.shape) == 2:
		pad_shape = (pad_h, pad_w)
	else:
		pad_shape = (pad_h, pad_w, (0, 0))
	img = np.pad(img, pad_shape, "constant", constant_values=cval)
	return img

def fix_mirror_padding(ann):
	"""Deal with duplicated instances due to mirroring in interpolation
	during shape augmentation (scale, rotation etc.).

	"""
	current_max_id = np.amax(ann)
	inst_list = list(np.unique(ann))
	inst_list.remove(0)  # 0 is background
	for inst_id in inst_list:
		inst_map = np.array(ann == inst_id, np.uint8)
		remapped_ids = measurements.label(inst_map)[0]
		remapped_ids[remapped_ids > 1] += current_max_id
		ann[remapped_ids > 1] = remapped_ids[remapped_ids > 1]
		current_max_id = np.amax(ann)
	return ann

def gen_instance_hv_map(ann, crop_shape):
	"""Input annotation must be of original shape.

	The map is calculated only for instances within the crop portion
	but based on the original shape in original image.
	Perform following operation:
	Obtain the horizontal and vertical distance maps for each
	nuclear instance.
	"""
	orig_ann = ann.copy()  # instance ID map
	fixed_ann = fix_mirror_padding(orig_ann)
	# re-cropping with fixed instance id map
	crop_ann = cropping_center(fixed_ann, crop_shape)
	# TODO: deal with 1 label warning
	crop_ann = morph.remove_small_objects(crop_ann, min_size=30)

	x_map = np.zeros(orig_ann.shape[:2], dtype=np.float32)
	y_map = np.zeros(orig_ann.shape[:2], dtype=np.float32)

	inst_list = list(np.unique(crop_ann))
	inst_list.remove(0)  # 0 is background
	for inst_id in inst_list:
		inst_map = np.array(fixed_ann == inst_id, np.uint8)
		inst_box = get_bounding_box(inst_map)

		# expand the box by 2px
		# Because we first pad the ann at line 207, the bboxes
		# will remain valid after expansion
		inst_box[0] -= 2
		inst_box[2] -= 2
		inst_box[1] += 2
		inst_box[3] += 2

		inst_map = inst_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]

		if inst_map.shape[0] < 2 or inst_map.shape[1] < 2:
			continue

		# instance center of mass, rounded to nearest pixel
		inst_com = list(measurements.center_of_mass(inst_map))

		inst_com[0] = int(inst_com[0] + 0.5)
		inst_com[1] = int(inst_com[1] + 0.5)

		inst_x_range = np.arange(1, inst_map.shape[1] + 1)
		inst_y_range = np.arange(1, inst_map.shape[0] + 1)
		# shifting center of pixels grid to instance center of mass
		inst_x_range -= inst_com[1]
		inst_y_range -= inst_com[0]

		inst_x, inst_y = np.meshgrid(inst_x_range, inst_y_range)

		# remove coord outside of instance
		inst_x[inst_map == 0] = 0
		inst_y[inst_map == 0] = 0
		inst_x = inst_x.astype("float32")
		inst_y = inst_y.astype("float32")

		# normalize min into -1 scale
		if np.min(inst_x) < 0:
			inst_x[inst_x < 0] /= -np.amin(inst_x[inst_x < 0])
		if np.min(inst_y) < 0:
			inst_y[inst_y < 0] /= -np.amin(inst_y[inst_y < 0])
		# normalize max into +1 scale
		if np.max(inst_x) > 0:
			inst_x[inst_x > 0] /= np.amax(inst_x[inst_x > 0])
		if np.max(inst_y) > 0:
			inst_y[inst_y > 0] /= np.amax(inst_y[inst_y > 0])

		####
		x_map_box = x_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
		x_map_box[inst_map > 0] = inst_x[inst_map > 0]

		y_map_box = y_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
		y_map_box[inst_map > 0] = inst_y[inst_map > 0]

	hv_map = np.dstack([x_map, y_map])
	return x_map, y_map, hv_map

def gen_targets(ann, crop_shape, **kwargs):
	"""Generate the targets for the network."""
	h_map, v_map, hv_map = gen_instance_hv_map(ann, crop_shape)
	np_map = ann.copy()
	np_map[np_map > 0] = 1

	h_map = cropping_center(h_map, crop_shape)
	v_map = cropping_center(v_map, crop_shape)
	hv_map = cropping_center(hv_map, crop_shape)
	np_map = cropping_center(np_map, crop_shape)

	target_dict = {
		"hv_map": hv_map,
		"h_map": h_map,
		"v_map": v_map,
		"np_map": np_map,
	}

	return target_dict

def prep_sample(data, is_batch=False, **kwargs):
	"""
	Designed to process direct output from loader
	"""
	cmap = plt.get_cmap("jet")

	def colorize(ch, vmin, vmax, shape):
		ch = np.squeeze(ch.astype("float32"))
		ch = ch / (vmax - vmin + 1.0e-16)
		# take RGB from RGBA heat map
		ch_cmap = (cmap(ch)[..., :3] * 255).astype("uint8")
		ch_cmap = center_pad_to_shape(ch_cmap, shape)
		return ch_cmap

	def prep_one_sample(data):
		shape_array = [np.array(v.shape[:2]) for v in data.values()]
		shape = np.maximum(*shape_array)
		viz_list = []
		viz_list.append(colorize(data["np_map"], 0, 1, shape))
		# map to [0,2] for better visualisation.
		# Note, [-1,1] is used for training.
		viz_list.append(colorize(data["hv_map"][..., 0] + 1, 0, 2, shape))
		viz_list.append(colorize(data["hv_map"][..., 1] + 1, 0, 2, shape))
		img = center_pad_to_shape(data["img"], shape)
		return np.concatenate([img] + viz_list, axis=1)

	# cmap may randomly fails if of other types
	if is_batch:
		viz_list = []
		data_shape = list(data.values())[0].shape
		for batch_idx in range(data_shape[0]):
			sub_data = {k : v[batch_idx] for k, v in data.items()}
			viz_list.append(prep_one_sample(sub_data))
		return np.concatenate(viz_list, axis=0)
	else:
		return prep_one_sample(data)

def get_hv_map(instance, input_image):
	# instance: truth inst_map int (1000, 1000)
	# input_image: the input image (1000, 1000, 3)

	target_dict = gen_targets(instance,[1000,1000])
	target_dict.pop('h_map')
	target_dict.pop('v_map')
	target_dict['img'] = input_image.reshape((1000,1000,3))
	hv_matrix = prep_sample(target_dict)
	horizontal = hv_matrix[:,2000:3000,:]
	vertical = hv_matrix[:,3000:,:]
	return horizontal, vertical
