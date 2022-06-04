import itertools
import random
import torch
import torchvision
import numpy as np

from fourier_heatmap import AddFourierNoise

mean = [0.49139968, 0.48215841, 0.44653091]
std  = [0.24703223, 0.24348513, 0.26158784]

class CustomNoise:
	# Class to create noise, every function needs eps which is v from (r*v*Ui,j) in the paper Yin et al. (2019)
	# h is height of the fourier_base
	# w is width of the fourier_base

	# Multiple frequencies are applied on one image with uniform
	def multiple_rand_uniform(self, eps, freqs: [(int, int)], num_of_transforms):
		transformations = []
		freqs_rand = random.sample(freqs, k=num_of_transforms)
		for (h, w) in freqs_rand:
			assert -16 <= h <= 15
			assert -16 <= w <= 15
			transformations.append(AddFourierNoise(h, w, (eps / num_of_transforms)))
		# print(transformations)
		return torchvision.transforms.Compose(transformations)

	# Applies one frequency at random from a range of h and w with probability p
	def random(self, eps, h_range: (int, int), w_range: (int, int), p=0.5):
		assert 0 <= p <= 1
		[h1, h2] = sorted(h_range)
		[w1, w2] = sorted(w_range)
		assert h1 >= -16 and h2 <= 15
		assert w1 >= -16 and w2 <= 15
		transformations = []
		for h in range(h1, h2 + 1):
			for w in range(w1, w2 + 1):
				transformations.append(AddFourierNoise(h, w, eps))
		# print(transformations)
		return torchvision.transforms.RandomChoice(transformations, [p] * len(transformations))

	# Returns the transformations based on error matrix
	# transformations are applied with probability p
	def error_metric(self, eps, error_matrix, error_rate=0.5, p=0.5):
		assert 0 <= p <= 1
		base_to_transform = []
		transformations = []
		for h in range(-16, 16):
			for w in range(-16, 16):
				if error_matrix[h, w] >= error_rate:
					base_to_transform.append((h, w))
		for (h, w) in base_to_transform:
			transformations.append(AddFourierNoise(h, w, eps))
		# print(transformations)
		return torchvision.transforms.RandomChoice(transformations, [p] * len(transformations))

	# Applies noise in either low, mid, or high frequency range, one frequency per image, random choice on which frequency
	def single_frequency_n(self, eps, mode: str, p:float=0.5, n:int=1):
		assert 0 <= p <= 1
		transformations = []
		comb_pos = []
		comb_neg = []
		if mode == "h":
			comb_pos = list(itertools.product(range(11, 16), repeat=2))
			comb_neg = list(map(lambda t: (t[0] * -1, t[1] * -1), comb_pos))
		elif mode == "m":
			comb_pos = list(itertools.product(range(6, 11), repeat=2))
			comb_neg = list(map(lambda t: (t[0] * -1, t[1] * -1), comb_pos))
		else:
			comb_pos = list(itertools.product(range(0, 6), repeat=2))
			comb_neg = list(map(lambda t: (t[0]*-1, t[1]*-1), comb_pos))
		comb = list(set(comb_pos + comb_neg))
		print(comb)
		for (h, w) in comb:
			transformations.append(AddFourierNoise(h, w, eps))
		return torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.RandomChoice(transformations, [p] * len(transformations)), torchvision.transforms.Normalize(mean, std)])

# noise = CustomNoise()
# er_matrix_example = torch.rand((4, 4))
# print(er_matrix_example)
# transform1 = noise.error_metric(2, er_matrix_example)


# Where -> which frequency do I apply (low, mid, high)
# How many -> how many frequencies per image
# Random? -> Is the selection of frequencies random or not
