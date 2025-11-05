from abc import ABC, abstractmethod
import numpy as np
import warnings

from typing import Optional, Any
from numpy.typing import NDArray

# for Gauss
from scipy.stats import multivariate_normal  # type: ignore


# Abstract class
class CorrelationMethod(ABC):
    # explicit annotation so subclasses' self.param_dict has a known type
    param_dict: dict[str, Any]

    @abstractmethod
    def make_time_offsets(
        self,
        shape_mask: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """
        Generate time offsets for the time of the shift. An offset value for each grid cell of the
        spatial grid is created at once.
        """
        pass


###################################################################################################
############################################# Uniform #############################################
###################################################################################################


class NoCorr(CorrelationMethod):
    """
    Default correlation method that does not apply any correlation. All offsets are zero.
    """

    def __init__(
        self,
    ):
        self.param_dict = {
            "correlation_type": NoCorr,
        }

    def make_time_offsets(
        self,
        shape_mask: NDArray[Any],
    ) -> NDArray[Any]:
        offsets = np.zeros(shape_mask.shape, dtype=int)

        return offsets


###################################################################################################
############################################# Noise ###############################################
###################################################################################################


class Noise(CorrelationMethod):
    def __init__(
        self,
        max_offset: int = 5,
        corr_seed: Optional[int] = None,
    ) -> None:
        self.max_offset = max_offset
        self.corr_seed = corr_seed

        self.param_dict = {
            "correlation_type": Noise,
            "max_offset": max_offset,
            "corr_seed": corr_seed,
        }

    def make_time_offsets(
        self,
        shape_mask: NDArray[Any],
    ) -> NDArray[Any]:
        if self.corr_seed is not None:
            rng = np.random.default_rng(self.corr_seed)
        else:
            rng = np.random.default_rng()
            warnings.warn(
                "No random seed set for PONDS Correlation Method. Results will not be reproducible."
            )

        # generate integer offsets directly to keep types explicit for static type checkers
        offsets = rng.integers(  # type: ignore
            -self.max_offset, self.max_offset, size=shape_mask.shape, dtype=int
        )

        return offsets  # type: ignore


###################################################################################################
############################################# Gauss ###############################################
###################################################################################################


class Gauss(CorrelationMethod):
    """
    Creates a Gauss-like correlation pattern. For a positive value `max_offset`, the central shift
    time occurs at the peak of the 2D gauss curve. All values around it are pushed towards the past.
    """

    def __init__(
        self,
        corr_center_lat: int | None = None,
        corr_center_lon: int | None = None,
        sigma_lat: int = 10,
        sigma_lon: int = 10,
        max_offset: int = 10,
    ) -> None:
        self.corr_center_lat = corr_center_lat
        self.corr_center_lon = corr_center_lon
        self.sigma_lat = sigma_lat
        self.sigma_lon = sigma_lon
        self.max_offset = max_offset

        self.param_dict = {
            "correlation_type": Gauss,
            "corr_center_lat": corr_center_lat,
            "corr_center_lon": corr_center_lon,
            "sigma_lat": sigma_lat,
            "sigma_lon": sigma_lon,
            "max_offset": max_offset,
        }

    def make_time_offsets(
        self,
        shape_mask: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        # Create lat/lon grid
        lats = np.linspace(-90, 90, shape_mask.shape[0])
        lons = np.linspace(-180, 180, shape_mask.shape[1])

        extended_lons = np.concatenate([lons - 360, lons, lons + 360])

        # Create grids
        lat_grid, extended_lon_grid = np.meshgrid(lats, extended_lons, indexing="ij")
        pos_extended = np.dstack((lat_grid, extended_lon_grid))

        # Calculate Gaussian over the whole extended grid
        cov = np.array([[self.sigma_lat**2, 0], [0, self.sigma_lon**2]])
        rv = multivariate_normal(
            mean=[self.corr_center_lat, self.corr_center_lon],
            cov=cov,  # type: ignore
        )
        extended_gauss = rv.pdf(pos_extended)  # type: ignore
        extended_gauss_scaled = (
            extended_gauss / np.max(extended_gauss) * self.max_offset
        )

        # Apply shift to data
        offsets = np.zeros(shape_mask.shape, dtype=int)
        for i in range(len(lats)):
            for j in range(len(lons)):
                # Sum contributions from periodic longitudes
                gauss_contribution = np.sum(
                    [extended_gauss_scaled[i, j + k * len(lons)] for k in range(3)],
                )
                offsets[i, j] = gauss_contribution - self.max_offset

        return offsets


###################################################################################################
############################################# Custom ##############################################
###################################################################################################


class Custom(CorrelationMethod):
    """
    Custom correlation method that allows for user-defined shift patterns.
    """

    def __init__(
        self,
        offset_mask: np.ndarray[Any, Any],
    ) -> None:
        self.offset_mask = offset_mask
        self.param_dict = {
            "correlation_type": Custom,
        }

    def make_time_offsets(
        self,
        shape_mask: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        # check if the offset_mask is of the correct shape
        if shape_mask.shape != self.offset_mask.shape:
            raise ValueError(
                f"Offset mask shape {self.offset_mask.shape} does not match grid shape {shape_mask.shape}."
            )
        # check if the offset_mask only contains integers
        if not np.issubdtype(self.offset_mask.dtype, np.integer):
            raise ValueError("Offset mask must be all integer values.")

        return self.offset_mask


###################################################################################################
########################################## Propagating ############################################
###################################################################################################


class Propagating(CorrelationMethod):
    """
    Mimics a propagating tipping event as proposed in Fig. 1 of [Loriani et al., 2025].

    Cluster mask is divided into Voronoi cells (VC). A random starting VC is picked, where the tipping event
    occurs first (offset=0). The further away a VC is from the starting VC, the later the tipping event occurs
    (larger offset value). The maximum offset value is determined by `max_offset`.
    """

    def __init__(
        self,
        n_cells: int,
        max_offset: int = 5,
        corr_seed: int | None = None,
    ) -> None:
        self.n_cells = n_cells
        self.max_offset = max_offset
        self.corr_seed = corr_seed

        self.param_dict = {
            "correlation_type": Propagating,
            "n_cells": n_cells,
            "max_offset": max_offset,
            "corr_seed": corr_seed,
        }

    def make_time_offsets(
        self,
        shape_mask: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """
        y = latitudes
        x = longitudes
        """

        if self.corr_seed is not None:
            rng = np.random.default_rng(self.corr_seed)
        else:
            rng = np.random.default_rng()
            warnings.warn(
                "No random seed set for PONDS Correlation Method. Results will not be reproducible."
            )

        if self.n_cells > np.sum(shape_mask):
            raise ValueError(
                f"Can not fit {self.n_cells=} cell centers into shape mask with {shape_mask.sum()} cells."
            )
        if self.n_cells > np.sum(shape_mask) * 0.5:
            print(
                f"Warning: Trying to draw {self.n_cells=} unique cell centers out of {shape_mask.sum()} possible cells. This may take a some time."
            )

        # get bounding box of shape_mask True values
        latmin_ix, latmax_ix = np.where(np.any(shape_mask, axis=0))[0][[0, -1]]
        lonmin_ix, lonmax_ix = np.where(np.any(shape_mask, axis=1))[0][[0, -1]]

        # draw random unique cell centers within shape_mask
        sx = np.full(self.n_cells, -1, dtype=int)
        sy = np.full(self.n_cells, -1, dtype=int)
        # ensure `pairs` is defined even if the loop does not run
        pairs = np.empty((0, 2), dtype=int)
        # - draw cell centers
        for i in range(self.n_cells):
            sx[i] = rng.integers(lonmin_ix, lonmax_ix, size=1)
            sy[i] = rng.integers(latmin_ix, latmax_ix, size=1)
            pairs = np.column_stack((sx[: i + 1], sy[: i + 1]))
            # - 1) check if drawn cell centers are not unique; 2) check if drawn cell center is not in shape_mask
            while (
                np.unique(pairs, axis=0).shape[0] < (i + 1)
                or not (shape_mask[sx[i], sy[i]])
            ):
                # - re-draw cell center
                sx[i] = rng.integers(lonmin_ix, lonmax_ix, size=1)
                sy[i] = rng.integers(latmin_ix, latmax_ix, size=1)
                pairs = np.column_stack((sx[: i + 1], sy[: i + 1]))
        self.cell_centers = pairs

        # assign each seed a threshold value spanning the max_offset range
        offset_val = self._offset_val_cells(rng)
        offset_mesh = np.array(
            [np.full(shape_mask.shape, offset_val[i]) for i in range(self.n_cells)]
        )

        # compute distance of each grid cell to each center
        yy, xx = np.mgrid[0 : shape_mask.shape[0], 0 : shape_mask.shape[1]]
        # compute squared distances for each cell center and ensure an explicit integer dtype
        d2: NDArray[np.int_] = np.stack(
            [
                (yy - self.cell_centers[i, 0]) ** 2
                + (xx - self.cell_centers[i, 1]) ** 2
                for i in range(self.n_cells)
            ],
            axis=0,
        ).astype(int)
        mask_offset_ix = np.argmin(d2, axis=0)
        # construct mask_offset array: use offset values at the indices of the closest cell centers
        mask_offset = np.array(
            [
                [
                    offset_mesh[mask_offset_ix[i, j], i, j]
                    for j in range(mask_offset_ix.shape[1])
                ]
                for i in range(mask_offset_ix.shape[0])
            ]
        )

        return mask_offset

    def _offset_val_cells(
        self,
        rng: np.random.Generator,
    ) -> NDArray[Any]:
        """
        Determines the offset values for each of the cells.

        Chose a random starting Voronoi cell, then assign offset values based on distance to that cell center.
        The further away from the starting cell center, the larger the offset value.
        """

        # chose a random index which will determine where the first tipping event occurs
        start_ix = rng.integers(0, self.n_cells - 1, size=1)
        # compute how close each cell center is to the starting index
        distance_to_start = np.array(
            [
                (self.cell_centers[i, 0] - self.cell_centers[start_ix, 0]) ** 2
                + (self.cell_centers[i, 1] - self.cell_centers[start_ix, 1]) ** 2
                for i in range(self.n_cells)
            ]
        )

        # assign offset values linearly increasing with distance to starting index
        offset_val = np.array(
            [
                int(distance_to_start[i] / distance_to_start.max() * self.max_offset)
                for i in range(self.n_cells)
            ]
        )

        return offset_val
