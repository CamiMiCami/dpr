import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from experiments import save_scatter_plot


class Constellation:
    def __init__(self, num_symbols, alpha, snr):
        """
        Args:
            num_symbols: number of constellation points
            alpha: amount of intersymbol interference
            snr: signal to noise ratio in dB
        """
        self.num_symbols = num_symbols
        self.alpha = alpha
        self.bits_per_symbol = np.log2(self.num_symbols)
        self.snr = None
        self.sigma = None
        self.set_snr(snr)

    def set_snr(self, snr):
        self.snr = snr
        # snr_linear = 1 / (2*sigma^2)
        # snr_dB = 10*log10(snr_linear)
        self.sigma = np.sqrt(0.5) * 10 ** (-self.snr / 20)

    def get_constellation_points(self):
        raise NotImplementedError

    def plot_constellation_points(self, label=False):
        """
        Visualize each normalized constellation point by a dot in the IQ-plane
        Args:
            label: annotate the symbols by labels
        """
        constellation_points = self.get_constellation_points()
        i, q = constellation_points[..., 0], constellation_points[..., 1]
        plt.scatter(i, q, c='black')
        plt.axis('square')

        if label:
            for n in range(self.num_symbols):
                plt.annotate(n, (i[n], q[n]), textcoords='offset pixels', xytext=(5, 5))

    @staticmethod
    def plot_samples(samples, alpha=0.01, save_plot=False):
        """
        Create a scatter plot of given samples.  Use this method to visualize noisy received samples
        in the IQ-plane
        Args:
            samples: samples of shape [number_of_samples, 2]
            alpha: transparency of scatters
            save_plot: save plot for usage together with pgfplots in publication
        """
        i, q = samples[..., 0], samples[..., 1]
        plt.scatter(i, q, alpha=alpha, c='black')
        if save_plot:
            save_scatter_plot('scatter_plot.png')

        else:
            plt.axis('square')

    def generate_symbols(self, number_samples):
        """
        Generate (noisy and with intersymbol interference) RX symbols
        Args:
            number_samples: number of RX samples to generate

        Returns:
            labels of the transmitted symbols (integer),
            noisy RX symbols of shape [number_of_samples, 2]
        """
        constellation_points = self.get_constellation_points()
        # sample a label (from uniform distribution because all labels are equally probable)
        labels = np.random.choice(self.num_symbols, number_samples)
        symbols_tx = constellation_points[labels]
        # AWGN: add i.i.d. Gaussian noise samples of zero mean and standard deviation sigma
        symbols_rx = symbols_tx + self.sigma * np.random.standard_normal(symbols_tx.shape)

        # apply FIR filter to model intersymbol interference
        symbols_rx = signal.lfilter([1., self.alpha], [1.], symbols_rx, axis=0)
        return labels, symbols_rx

    def get_discriminant_function_values(self, samples):
        """
        Determine the values of each class' discriminant function
        Args:
            samples: RX samples

        Returns:
            discriminant function values, array of shape [num_samples, num_constellation_points]
        """
        constellation_points = self.get_constellation_points()

        # if no intersymbol interference (i.e. alpha = 0) simply Gaussian distribution
        # otherwise multimodal likelihood, store mean value of each component of the Gaussian mixture
        # model in an array of shape [number_of_symbols, number_of_components, 2]
        if self.alpha:
            repeats = self.num_symbols
        else:
            repeats = 1
        means = np.repeat(constellation_points[:, np.newaxis, :], repeats=repeats, axis=1)
        if self.alpha:
            means += constellation_points * self.alpha
        means = means[np.newaxis, ...]

        # reshape samples such that we can use broadcasting of numpy
        samples = np.reshape(samples, [-1, 1, 1, 2])
        disc_function_values = -np.sum((samples - means)**2, axis=-1)
        disc_function_values = np.sum(disc_function_values, axis=-1)
        return disc_function_values

    def estimate_label(self, samples):
        """
        Estimate label given RX samples
        Args:
            samples: RX samples

        Returns:
            vector of integers representing the estimated label of each RX sample
        """
        disc_function_values = self.get_discriminant_function_values(samples)
        # symbol with highest discriminant value is chosen
        estimated_labels = np.argmax(disc_function_values, axis=-1)

        return estimated_labels

    def determine_symbol_error_rate(self, number_samples):
        """
        Estimate the symbol error rate of the constellation using Monte Carlo Simulation
        Args:
            number_samples: number of samples to use for simulation

        Returns:
            estimated symbol error rate
        """
        number_samples = int(number_samples)
        labels, symbols_rx = self.generate_symbols(number_samples)
        estimated_labels = self.estimate_label(symbols_rx)
        # estimate symbol error rate using Monte Carlo
        symbol_error_rate = np.sum(np.not_equal(labels, estimated_labels)) / number_samples
        return symbol_error_rate

    def determine_bit_error_rate(self, number_samples):
        """
        Estimate the bit error rate of the constellation using Monte Carlo Simulation
        Args:
            number_samples: number of samples to use for simulation

        Returns:
            estimated bit error rate
        """
        # bit error rate P_b and symbol error rate P_s are related through P_s = M_b * P_b with
        # M_b = bits per symbol
        bit_error_rate = self.determine_symbol_error_rate(number_samples) / self.bits_per_symbol
        return bit_error_rate


class QAMConstellation(Constellation):
    def __init__(self, num_symbols, alpha, snr):
        super(QAMConstellation, self).__init__(num_symbols, alpha, snr)

        self.name = f"{num_symbols}-QAM (alpha = {alpha})"

        # number of constellation points must be a perfect square
        if np.not_equal((self.num_symbols ** 0.5) % 1, 0):
            raise NotImplementedError("number of constellation points must be a perfect square")

    def get_constellation_points(self):
        """
        Determine the normalized constellation points of the specified QAM.
        Returns:
            normalized constellation poinst, an array of shape [number_of_constellation_points, 2]
        """
        symbols_per_component = int(np.sqrt(self.num_symbols))
        # equally spaced real and imaginary components of the symbols
        levels = np.arange(symbols_per_component)
        i, q = np.meshgrid(levels, levels)
        symbols = np.stack([i, q], axis=-1)
        mean = np.mean(symbols)
        # constellation is symmetric with respect to real and imaginary axis
        symbols = symbols - mean
        # normalize average power
        avg_power = np.mean(np.sum(symbols**2, axis=-1))
        symbols = symbols / np.sqrt(avg_power)
        symbols = np.reshape(symbols, (-1, 2))
        return symbols


class PSKConstellation(Constellation):
    def __init__(self, num_symbols, alpha, snr):
        super(PSKConstellation, self).__init__(num_symbols, alpha, snr)

        self.name = f"{num_symbols}-PSK (alpha = {alpha})"

    def get_constellation_points(self):
        """
        Determine the normalized constellation points of the specified PSK.
        Returns:
            normalized constellation poinst, an array of shape [number_of_constellation_points, 2]
        """
        # equally spaced angles
        angles = 2 * np.pi * np.arange(self.num_symbols) / self.num_symbols
        i = np.cos(angles)
        q = np.sin(angles)
        symbols = np.stack([i, q], axis=-1)
        return symbols
