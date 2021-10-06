import numpy as np
from abc import ABC, abstractmethod
import scipy.stats as ss
import matplotlib.pyplot as plt


class DataDistribution(ABC):

    def __init__(self, dim):
        self.dim = dim

    @abstractmethod
    def sample(self, n_samples):
        pass

    @abstractmethod
    def visualize(self):
        pass


class UniformDistribution(DataDistribution):
    def __init__(self, dim, lb, ub):
        super(UniformDistribution, self).__init__(dim)
        self.sampler = np.random.uniform
        self.lb = lb
        self.ub = ub

    def sample(self, n_samples):
        return self.sampler(self.lb, self.ub, size=(n_samples, self.dim))

    def visualize(self):
        x = np.linspace(self.lb, self.ub, 200)
        y = ss.uniform.pdf(x, self.lb, self.ub - self.lb)
        plt.plot(x, y)
        plt.xlabel('Random variable')
        plt.ylabel('pdf')
        plt.show()


class PolytopicUniformDistribution(DataDistribution):
    def __init__(self, dim, lb, ub, A, b):
        super(PolytopicUniformDistribution, self).__init__(dim)
        self.lb = lb
        self.ub = ub
        self.sampler = np.random.uniform
        self.A = A
        self.b = b

    def sample(self, n_samples):
        samples = []
        while len(samples) < n_samples:
            sample = self.sampler(self.lb, self.ub, size=self.dim)
            if np.all(self.A*sample <= self.b):
                samples.append(sample)

        return np.vstack(samples)

    def visualize(self):
        raise NotImplementedError


class GaussianDistribution(DataDistribution):
    def __init__(self, dim, mean, std):
        super(GaussianDistribution, self).__init__(dim)
        self.mean = mean
        self.std = std
        self.sampler = np.random.multivariate_normal
        # self.sampler = np.random.normal

    def sample(self, n_samples):
        return self.sampler(self.mean, self.std, size=n_samples)
        # return self.sampler(loc=self.mean, scale=self.std, size=(n_samples, self.dim))

    def visualize(self):
        x = np.linspace(self.mean - 2 * self.std, self.mean + 2 * self.std, 200)
        y = ss.norm.pdf(x)
        plt.plot(x, y)
        plt.xlabel('Random variable')
        plt.ylabel('pdf')
        plt.show()


class PolytopicGaussianDistribution(DataDistribution):
    def __init__(self, dim, mean, std, A, b):
        super(PolytopicGaussianDistribution, self).__init__(dim)
        self.mean = mean
        self.std = std
        self.sampler = np.random.multivariate_normal
        self.A = A
        self.b = b

    def sample(self, n_samples):
        samples = []
        while len(samples) < n_samples:
            sample = self.sampler(self.mean, self.std)
            if np.all(self.A*sample <= self.b):
                samples.append(sample)

        return np.vstack(samples)

    def visualize(self):
        raise NotImplementedError


class GaussianMixtureDistribution(DataDistribution):
    def __init__(self, dim, means, stds, weights):
        super(GaussianMixtureDistribution, self).__init__(dim)
        self.means = np.asarray(means)
        self.stds = np.asarray(stds)
        assert np.sum(weights) == 1, "Weights do not sum up to 1!"
        self.weights = weights
        self.n_components = len(weights)
        self.sampler = np.random.multivariate_normal

    def sample(self, n_samples):
        mixture_idx = np.random.choice(len(self.weights),
                                       size=n_samples,
                                       p=self.weights)
        samples = np.zeros((n_samples, self.dim))
        for i, idx in enumerate(mixture_idx):
            samples[i, :] = self.sampler(self.means[idx],
                                         self.stds[idx])
        return samples

    def visualize(self):
        x_low = np.min(self.means - 2 * self.stds)
        x_high = np.max(self.means + 2 * self.stds)
        x = np.linspace(x_low, x_high, 200)
        y = np.zeros_like(x)
        for i, w in enumerate(self.weights):
            y += ss.norm.pdf(x, loc=self.means[i], scale=self.stds[i]) * w
        plt.plot(x, y)
        plt.xlabel('Random variable')
        plt.ylabel('pdf')
        plt.show()
        pass
