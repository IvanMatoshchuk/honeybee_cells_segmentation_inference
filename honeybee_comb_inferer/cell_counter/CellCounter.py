import cv2
import time
from typing import Tuple
from scipy import ndimage
from skimage.feature import peak_local_max

import numpy as np


class CellCounter:
    """
    Class for counting individual cells from the inferred mask of the comb without bees.
    This mask can be produced by filming comb for a short amount of time
    and applying 'infer_without_bees' from HoneyBeeCombInferrer class

    Parameters
    ----------


    """

    def __init__(self, inferred_mask: np.ndarray, method: str = "edt"):
        self.inferred_mask = inferred_mask
        self.method = method

    def run_counter(self):
        if self.method == "edt":
            output = self._run_edt()
        elif self.method == "cht":
            output = self._run_cht()
        else:
            raise Exception(
                f"Method <{self.method}> is not defined!"
                "Either Euclidean Distance Transform ('edt') or Circle Hough Transform ('cht') method should be used!"
            )
        return output

    def _run_edt(self) -> dict:
        output = {}
        start = time.time()
        total_num_cells = 0
        for label in np.unique(self.inferred_mask)[1:]:
            temp_mask = self.inferred_mask.astype("uint8").copy()

            distance_map, thresh = self._get_distance_map_and_binarized_image(mask=temp_mask, label=label)

            local_max = peak_local_max(distance_map, min_distance=35, labels=thresh)
            output[label] = local_max.shape[0]
            total_num_cells += local_max.shape[0]
        end = time.time()
        print("Time taken: ", f"{round(start - end, 3)} sec.")
        print(f"Total number of cells: {total_num_cells}")
        return output

    def _run_cht(self):
        pass

    def _get_distance_map_and_binarized_image(self, mask: np.ndarray, label: int) -> Tuple[np.ndarray, np.ndarray]:

        mask[mask != label] = 0
        thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        distance_map = ndimage.distance_transform_edt(thresh)

        return distance_map, thresh
