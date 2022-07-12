import torchvision

from fourier_heatmap import AddFourierNoise


class CustomNoise:
    """Class to create noise, every function needs eps which is v from (r*v*Ui,j) in the paper by Yin et al. (2019)
    h is height of the fourier_base
    w is width of the fourier_base
    the weighting can be changed by instead of giving a number, giving a list that can be used. However, this needs to
    be implemented, so for now it does not matter what p is as it is uniformly random
    """

    def freq_layered_range(self, eps_list, f_range: str):
        """Creates layers of noise, indicated by a string containing the frequency range
        :param eps_list: in order list of strengths for the frequency layer
        :param f_range: string that contains the char for the frequency range "l" for low, "m" for mid and "h" for high;
        it can be combined as for example "hl" which would layer high and low frequency noise
        :return: list of transformations
        """
        transformations = []
        ranges = list(f_range)
        assert len(eps_list) == len(ranges)
        for i, r in enumerate(ranges):
            transformations.append(self.single_frequency(eps_list[i], ranges[i], 0.5))
        return transformations

    def error_metric(self, eps, error_matrix, error_rate=0.5, p: float = 0.5, further_processing: bool = False):
        """Returns the transformations based on error matrix transformations are applied with probability p
        :param error_matrix: the error matrix extracted from the create_fourier_heatmap function
        :param error_rate: consider the frequencies as noise above the error rate
        :param p: probability of choosing the transformation for an image (only applicable if further_processing is False)
        :param further_processing: if set then transformation list is returned for further processing with different probabilities
        :return: a transformation that chooses a high, mid, or low frequency randomly at probability p or just the transformations
        """
        assert 0 <= p <= 1
        transformations = []
        for h in range(-16, 16):
            for w in range(-16, 16):
                if error_matrix[h, w] >= error_rate:
                    transformations.append(AddFourierNoise(h, w, eps))
        if further_processing:
            return transformations
        return torchvision.transforms.RandomChoice(transformations, [p] * len(transformations))

    def single_frequency(self, eps, mode: str, p: float = 0.5, further_processing: bool = False):
        """Applies noise in either low, mid, or high frequency range, random choice with probability p on which frequency
        Low, Mid, and High frequency ranges have been defined in 3 (almost) evenly identical sized sets
        :param eps: strength of noise, which is equal for every transformation, if different strengths are required then pleaser refer to single_frequency_w
        :param mode: mode must be either h = high, m = mid, or l = low for the respective frequency ranges
        :param p: probability of choosing the transformation for an image (only applicable if further_processing is False)
        :param further_processing: if set then transformation list is returned for further processing with different probabilities
        :return: a transformation that chooses a high, mid, or low frequency randomly at probability p or just the transformations
        """
        assert 0 <= p <= 1
        transformations = []
        h = [-16, -15, -14, -13, -12, 11, 12, 13, 14, 15]
        m = [-11, -10, -9, -8, -7, -6, 5, 6, 7, 8, 9, 10]
        l = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]
        comb = []
        if mode == "h":
            for i in h:
                for j in range(-16, 16):
                    comb.append((i, j))
                    if i != j:
                        comb.append((j, i))
        elif mode == "m":
            for i in m:
                for j in range(-11, 11):
                    comb.append((i, j))
                    if i != j:
                        comb.append((j, i))
        elif mode == "l":
            for i in l:
                for j in range(-5, 5):
                    comb.append((i, j))
                    if i != j:
                        comb.append((j, i))
        else:
            raise Exception
        for (h, w) in comb:
            transformations.append(AddFourierNoise(h, w, eps))
        if further_processing:
            return transformations
        return torchvision.transforms.RandomChoice(transformations, [p] * len(transformations))

    # takes transformation coordinates and creates the corresponding Fourier-Basis noise with strength eps
    def custom_transform(self, transform_coord, eps):
        transforms = []
        for (h, w) in transform_coord:
            transforms.append(AddFourierNoise(h, w, eps))
        return transforms
