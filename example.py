from pathlib import Path

import matplotlib.pyplot as plt
from skimage.transform import rescale

from radon_tomography import Radon


def main():
    IMAGES_PATH = Path("images")

    image = plt.imread(IMAGES_PATH / "shepp_logan.jpg")
    image = rescale(image, scale=0.4, mode="reflect", channel_axis=None)

    radon_params = {
        "num_angles": 320,
        "min_angle": 0,
        "max_angle": 180,
    }

    radon_transform = Radon(
        image,
        num_angles=radon_params["num_angles"],
        min_angle=radon_params["min_angle"],
        max_angle=radon_params["max_angle"],
    )

    radon_transform.compute_sinogram()
    radon_transform.invert_sinogram(filter_name="ramlak")

    plot_results(image, radon_transform)


def plot_results(original_image, radon_transform):
    plt.figure(figsize=(10, 7))

    # Plot original image
    plt.subplot(131)
    plt.imshow(original_image, cmap="gray")
    plt.title("Original Image")

    # Plot sinogram
    plt.subplot(132)
    plt.imshow(radon_transform.sinogram, cmap="gray")
    plt.title("Sinogram")

    # Plot reconstructed image
    plt.subplot(133)
    plt.imshow(radon_transform.inverted_sinogram, cmap="gray")
    plt.title("Reconstructed Image")

    plt.show()


if __name__ == "__main__":
    main()
