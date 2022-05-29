import time
from typing import Tuple, Union, List

import cv2
import numpy as np
import skimage
from scipy import ndimage
from skimage.feature import peak_local_max
from honeybee_comb_inferer.config import label_classes_default
from honeybee_comb_inferer.utils.utils import get_label_class_mapping


class CellCounter:
    """
    Class for counting individual cells from the inferred mask of the comb without bees.
    This mask can be produced by filming comb for a short amount of time
    and applying 'infer_without_bees' from HoneyBeeCombInferrer class

    Parameters
    ----------
        inferred_mask: np.ndarray
            output of HoneyBeeInferer.infer_without_bees. Mask made from images filmed in short time.
        method: str
            method to use for counting of cells. Either 'edt' (Euclidean Distance Transform) or 'cht' (Circle Hough Transform).
    """

    def __init__(
        self,
        inferred_mask: np.ndarray,
        method: str = "edt",
        label_classes_config: Union[str, List[dict]] = label_classes_default,
    ):
        self.inferred_mask = inferred_mask
        self.method = method
        self.label_classes_mapping = get_label_class_mapping(label_classes_config)

    def run_counter(self) -> dict:
        """
        main function for counting individual cells

        Output
        ------
            output: dict
                dictionary containing label -> count mapping.
        """
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
            output[self.label_classes_mapping[label]] = local_max.shape[0]
            total_num_cells += local_max.shape[0]
        end = time.time()
        print("Time taken: ", f"{round(end - start, 3)} sec.")
        print(f"Total number of cells: {total_num_cells}")
        return output

    def _run_cht(self) -> dict:
        output = {}
        start = time.time()
        total_num_cells = 0

        # hard-coded radious, since distance to camera is the same for all images
        radius = 48.0 / 2.0
        radius_range = np.arange(int(radius * 0.75), int(radius * 1.25))

        for label in np.unique(self.inferred_mask)[1:]:
            temp_mask = self.inferred_mask.astype("uint8").copy()

            cell_borders, thresh = self._get_cell_border_and_binarized_image(mask=temp_mask, label=label)
            hough = skimage.transform.hough_circle(cell_borders, radius=radius_range)
            accum, cx, cy, rad = skimage.transform.hough_circle_peaks(
                hough,
                radii=radius_range,
                min_xdistance=int(1.5 * radius),
                min_ydistance=int(1.5 * radius),
                normalize=True,
            )

            output[self.label_classes_mapping[label]] = cx.shape[0]
            total_num_cells += cx.shape[0]
        end = time.time()
        print("Time taken: ", f"{round(end - start, 3)} sec.")
        print(f"Total number of cells: {total_num_cells}")
        return output

    def _get_distance_map_and_binarized_image(self, mask: np.ndarray, label: int) -> Tuple[np.ndarray, np.ndarray]:

        mask[mask != label] = 0
        thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        distance_map = ndimage.distance_transform_edt(thresh)

        return distance_map, thresh

    def _get_cell_border_and_binarized_image(self, mask: np.ndarray, label: int) -> Tuple[np.ndarray, np.ndarray]:

        mask[mask != label] = 0
        thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        cell_borders = skimage.feature.canny(thresh)

        return cell_borders, thresh
