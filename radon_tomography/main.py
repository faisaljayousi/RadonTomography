import numpy as np
from scipy.ndimage import rotate
from .filter_factory import FilterFactory


class Radon:

    def __init__(self, input_image, num_angles, min_angle, max_angle):
        """
        Initialises Radon class with input image and angle parameters.

        Args:
            input_image (numpy.ndarray): The 2D image to compute the sinogram from.
            num_angles (int): Number of projection angles to use.
            min_angle (float): The minimum angle for projections (in degrees).
            max_angle (float): The maximum angle for projections (in degrees).
        """

        if not (0 <= min_angle < max_angle <= 360):
            raise ValueError(
                "min_angle must be less than max_angle and both within [0, 360]."
            )

        self.image = input_image
        self.num_angles = num_angles
        self.min_angle = min_angle
        self.max_angle = max_angle

        self.sinogram = None
        self.inverted_sinogram = None

        self.image_size = max(*input_image.shape)

        self.angles = np.linspace(
            min_angle, max_angle * (1 - 1 / num_angles), num_angles
        )

    def compute_sinogram(self):
        """
        Computes the sinogram of the input image by summing the line integrals
        at specified projection angles.
        """
        self.sinogram = np.zeros((self.num_angles, self.image_size))
        for idx, angle in enumerate(self.angles):
            self.compute_line_integral(angle, idx)
        self.sinogram = np.transpose(self.sinogram)

    def invert_sinogram(self, filter_name=None):
        """
        Inverts the sinogram using filtered back projection (FBP)
        with an optional filter. If no filter is provided, simple
        back projection is applied.

        Args:
            filter_name (str or None): The name of the filter to
            be applied. If None, simple back projection is performed.
        """
        if self.sinogram is None:
            raise RuntimeError(
                "Must call `compute_sinogram()` before inverting."
            )

        self.inverted_sinogram = np.zeros(
            (self.image_size, self.image_size), dtype=self.sinogram.dtype
        )

        frequency_filter = FilterFactory.create_filter(
            filter_name, self.image_size
        )

        for index, angle in enumerate(self.angles):
            projection = self.sinogram[:, index]
            filtered_projection = self.fbp(projection, frequency_filter)
            repeated_projection = np.tile(
                filtered_projection, (self.image_size, 1)
            )
            self.inverted_sinogram += rotate(
                repeated_projection, -angle, reshape=False
            )

    def compute_line_integral(self, theta, idx):
        """Computes a single line integral at a given angle theta."""
        rotated_image = rotate(self.image, theta, reshape=False)
        self.sinogram[idx] = np.sum(rotated_image, axis=0)

    def fbp(self, projection, frequency_filter):
        """
        Applies a filter in the frequency domain to a projection.

        Args:
            projection (numpy.ndarray): The projection to be filtered.
            frequency_filter (numpy.ndarray): The filter to apply in the
            frequency domain.

        Returns:
            numpy.ndarray: The filtered projection.
        """
        projection_fft = np.fft.fft(projection)
        projection_fft_shifted = np.fft.fftshift(projection_fft)
        filtered_projection_fft = projection_fft_shifted * frequency_filter
        filtered_projection_fft = np.fft.fftshift(filtered_projection_fft)
        return np.fft.ifft(filtered_projection_fft).real
