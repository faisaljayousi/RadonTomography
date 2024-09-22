import numpy as np


class FilterFactory:
    @staticmethod
    def create_filter(filter_name, N):
        w = np.linspace(-1, 1, N)

        if filter_name == "ramlak":
            return np.abs(w)
        elif filter_name == "shepp-logan":
            return np.abs(w) * np.sinc(w / 2)
        elif filter_name == "cosine":
            return np.abs(w) * np.cos(np.pi * w / 2)
        elif filter_name is None:
            return w
        else:
            raise ValueError(f"Unknown filter: {filter_name}")
