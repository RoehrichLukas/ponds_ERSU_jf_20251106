from abc import ABC, abstractmethod
import numpy as np
import warnings
import copy

from typing import Any, Optional, Union
from numpy.typing import NDArray

# for BLOB
from scipy.ndimage import label  # type: ignore


# Abstract class
class ShapeMethod(ABC):
    # explicit annotation so subclasses' self.param_dict has a known type
    param_dict: dict[str, Any]

    @abstractmethod
    def make_shape(
        self,
        lats: NDArray[np.float64],
        lons: NDArray[np.float64],
    ) -> NDArray[np.bool_]:
        raise NotImplementedError


###################################################################################################
############################################# SINGLE ##############################################
###################################################################################################


class SINGLE(ShapeMethod):
    def __init__(self) -> None:
        """
        A placeholder for a single shape method. This is used when no specific shape method is defined.
        """

        self.param_dict = {
            "shape_type": SINGLE,
        }

    def make_shape(
        self,
        lats: NDArray[np.float64],
        lons: NDArray[np.float64],
    ) -> NDArray[np.bool_]:
        return np.ones(
            (len(lats), len(lons)), dtype=bool
        )  # Single shape covering the entire grid


###################################################################################################
############################################### CELL ##############################################
###################################################################################################


class CELL(ShapeMethod):
    def __init__(
        self,
        center_lat: Optional[float] = None,
        center_lon: Optional[float] = None,
    ) -> None:
        if center_lat is None:
            center_lat = np.random.randint(-80, 80)
        if center_lon is None:
            center_lon = np.random.randint(-180, 180)

        self.center_lat = center_lat
        self.center_lon = center_lon

        self.param_dict = {
            "shape_type": CELL,
            "center_lat": center_lat,
            "center_lon": center_lon,
        }

    def make_shape(
        self,
        lats: NDArray[np.float64],
        lons: NDArray[np.float64],
    ) -> NDArray[np.bool_]:
        """
        Generates a 2D spatial cell shape. Given are the central latitude and longitude.
        The function returns a mask of the same shape as the grid, where only the cell at the specified
        latitude and longitude is True, and all other values are False.
        """

        lat_idx = np.argmin(np.abs(lats - self.center_lat))
        lon_idx = np.argmin(np.abs(lons - self.center_lon))
        shape_mask = np.zeros((len(lats), len(lons)), dtype=bool)
        shape_mask[lat_idx, lon_idx] = True

        return shape_mask


###################################################################################################
######################################## ELLIPSIS #################################################
###################################################################################################


class ELLIPSIS(ShapeMethod):
    def __init__(
        self,
        center_lat: Optional[float] = None,  # center latitude degree of the ellipsis
        center_lon: Optional[float] = None,  # center longitude degree of the ellipsis
        ecc_lat: Optional[float] = None,  # standard deviation in latitude degrees
        ecc_lon: Optional[float] = None,  # standard deviation in longitude degrees
        shape_seed: Optional[int] = None,
    ) -> None:
        if shape_seed is not None:
            # np.random.seed(random_seed)
            rng = np.random.default_rng(shape_seed)
        else:
            rng = np.random

        if center_lat is None:
            center_lat = rng.uniform(-80, 80)
        if center_lon is None:
            center_lon = rng.uniform(-180, 180)
        if ecc_lat is None:
            ecc_lat = rng.uniform(10, 20)
        if ecc_lon is None:
            ecc_lon = rng.uniform(15, 40)

        self.center_lat = center_lat
        self.center_lon = center_lon
        self.ecc_lat = ecc_lat
        self.ecc_lon = ecc_lon

        self.param_dict = {
            "shape_type": ELLIPSIS,
            "center_lat": center_lat,
            "center_lon": center_lon,
            "ecc_lat": ecc_lat,
            "ecc_lon": ecc_lon,
        }

    def make_shape(
        self,
        lats: NDArray[np.float64],
        lons: NDArray[np.float64],
    ) -> NDArray[np.bool_]:
        """
        Generates a 2D spatial ellipsis shape. Given are the central latitude and longitude, as well as the
        extend of the ellipsis in latitude and longitude. The function returns a mask of the same shape as the grid,
        where are values that are part of the ellipsis are True, and all other values are False.

        Elliptic Formula:
        ((lat - center_lat) / sigma_lat)**2 + ((lon - center_lon) / sigma_lon)**2 <= 1
        """

        # Create coordinate combinations of all latitudes and longitudes
        # - handle longitude periodicity to ensure continuity at the dateline (global dataset!)
        extended_lons = np.concatenate([lons - 360, lons, lons + 360])
        # - create meshgrid for latitudes and extended longitudes
        lat_grid, lon_grid = np.meshgrid(lats, extended_lons, indexing="ij")

        # Calculate the mask for the elliptic shape on the longitude-extended grid
        shape_mask = ((lat_grid - self.center_lat) / self.ecc_lat) ** 2 + (
            (lon_grid - self.center_lon) / self.ecc_lon
        ) ** 2 <= 1

        # add up contributions from periodic longitudes, shape_contribution is of shape (lat_size, lon_size)
        # - split shape_mask into three parts, one for each longitude section
        shape_mask = shape_mask.reshape(len(lats), 3, len(lons))
        # - transpose to have latitudes as first dimension
        shape_mask = shape_mask.transpose(1, 0, 2)
        # - if any of the three sections is True, then the point is part of the shape
        shape_mask = np.any(shape_mask, axis=0)

        return shape_mask


###################################################################################################
########################################### CUSTOM ################################################
###################################################################################################


class CUSTOM(ShapeMethod):
    def __init__(self, shape_mask: NDArray[np.bool_]) -> None:
        """
        Custom shape method that uses a pre-defined mask.

        Args:
            shape_mask (np.ndarray): A 2D boolean array of shape (lat_size, lon_size) representing
            the shape mask.
        """

        self.shape_mask = np.array(shape_mask)
        self.param_dict = {
            "shape_type": CUSTOM,
        }

    def make_shape(
        self,
        lats: NDArray[np.float64],
        lons: NDArray[np.float64],
    ) -> NDArray[np.bool_]:
        """
        Returns the pre-defined shape mask. Checks if the mask is of the correct shape and type.

        Returns:
            shape_mask (np.ndarray): The pre-defined shape mask.
        """

        # check if the shape_mask is of the correct shape
        if self.shape_mask.shape != (len(lats), len(lons)):
            raise ValueError(
                f"Shape mask shape {self.shape_mask.shape} does not match grid shape {(len(lats), len(lons))}."
            )
        # check if the shape_mask is a boolean array
        if not np.issubdtype(self.shape_mask.dtype, np.bool_):
            raise ValueError("Shape mask must be a boolean array.")

        return self.shape_mask


###################################################################################################
############################################ BLOB #################################################
###################################################################################################


class BLOB(ShapeMethod):
    def __init__(
        self,
        center_lat: Optional[float] = None,
        center_lon: Optional[float] = None,
        size: Optional[float] = None,
        max_bellys: Optional[int] = None,
        goo: Optional[float] = None,
        shape_seed: Optional[int] = None,
    ) -> None:
        """
        Creates a shape using the following algorithm:
            - place `max_bellys` points in proximity to the center point
            - draw a circle around each point with a radius around `size`
            - apply a water-surface effect depending on the strength of `goo` (like stickyness)
                - if `goo` is not None, the algorithm will find the largest `goo` value for
                    which all circles are still connected

        Args:
            center_lat (float): The latitude of the center point.
            center_lon (float): The longitude of the center point.
            size (float): The size of the circles.
            max_bellys (int): The maximum number of points to place.
            goo (float): 'Stickyness' of the water-surface effect.
            shape_seed (int): Seed for reproducibility.
        """

        if shape_seed is not None:
            rng = np.random.default_rng(shape_seed)
        else:
            rng = np.random.default_rng()

        if center_lat is None:
            center_lat = rng.uniform(-80, 80)
        if center_lon is None:
            center_lon = rng.uniform(-180, 180)
        if size is None:
            size = rng.uniform(5, 10)
        if max_bellys is None:
            max_bellys = rng.integers(2, 10)

        self.center_lat = center_lat
        self.center_lon = center_lon
        self.size = size
        self.max_bellys = max_bellys
        self.goo = goo
        self.random_seed = shape_seed

        self.param_dict = {
            "shape_type": BLOB,
            "center_lat": center_lat,
            "center_lon": center_lon,
            "size": size,
            "max_bellys": max_bellys,
            "goo": goo,
            "shape_seed": shape_seed,
        }

    def make_shape(
        self,
        lats: NDArray[np.float64],
        lons: NDArray[np.float64],
    ) -> NDArray[np.bool_]:
        """Create a metaball mask in lat/lon space."""

        if self.random_seed is not None:
            rng = np.random.default_rng(self.random_seed)
        else:
            rng = np.random.default_rng()
            warnings.warn(
                "No random seed set for PONDS Shape Method. Results will not be reproducible."
            )

        # Create a few metaballs near the center
        balls: list["_Ball"] = []
        # place first ball in center
        balls.append(_Ball(self.center_lat, self.center_lon, size=self.size))
        for _ in range(1, self.max_bellys):
            offset = 5 * self.size
            lat_offset = rng.uniform(-offset, offset)
            lon_offset = rng.uniform(-offset, offset)
            size = self.size / self.max_bellys * rng.uniform(0.8, 1.2)
            balls.append(
                _Ball(
                    self.center_lat + lat_offset,
                    self.center_lon + lon_offset,
                    size=size,
                )
            )

        # Finding the right value for `goo`
        # - 1st Case: Single metaball, blob will always be connected
        if self.goo is None and self.max_bellys == 1:
            self.goo = 2.0
        # - 2nd Case: Multiple metaballs, need to find optimal goo
        elif self.goo is None and self.max_bellys != 1:
            step = 0.1
            goo = 1.5  # low stickyness leads to larger expansion of the blob
            mbs = _Metaball(balls, goo=goo, threshold=0.02)
            mask = mbs.field_mask(lats, lons)
            # check if all TRUE in mask are still connected
            while self._mask_connected(mask) and goo < 4.0:
                goo += step
                mbs = _Metaball(balls, goo=goo, threshold=0.02)
                mask = mbs.field_mask(lats, lons)
            # mask is not connected -> reduce goo again
            goo -= step  # last goo was too much
            self.goo = goo

        # goo is set, create metaball system
        goo_value = self.goo if self.goo is not None else 2.0
        mbs = _Metaball(balls, goo=goo_value, threshold=0.02)
        shape_mask = mbs.field_mask(lats, lons)

        self.param_dict["goo"] = self.goo

        return shape_mask

    def _mask_connected(
        self,
        mask: NDArray[np.bool_],
    ) -> bool:
        """
        Check if all TRUE values of the mask are connected to each other.
        """
        if not np.any(mask):
            return False  # No True cells at all

        # connectivity structure
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

        # Label connected components in extended mask
        _, num_features = label(mask, structure=structure)  # type: ignore # use scipy.ndimage function

        if num_features == 1:
            return True
        # check if split at longitude
        else:
            lon_size = mask.shape[1]
            extended_mask = np.hstack([mask, mask])  # (lat, 2*lon)
            extended_mask = extended_mask[
                :, lon_size // 2 : 3 * lon_size // 2
            ]  # keep only the middle section

            # Label connected components in extended mask
            _, num_features = label(  # type: ignore
                extended_mask, structure=structure
            )  # type: ignore # use scipy.ndimage function

            if num_features == 1:
                return True
            else:
                return False


# Helper Classes for the Blob Generation
class _Ball:
    """Single metaball."""

    def __init__(
        self,
        lat: float,
        lon: float,
        size: float,
    ) -> None:
        self.lat = lat
        self.lon = lon
        self.size = size


class _Metaball:
    """Metaball system in lat/lon space with longitude wrapping."""

    def __init__(
        self,
        balls: list[_Ball],
        goo: float = 2.0,
        threshold: float = 0.0004,
    ) -> None:
        self.balls = balls
        self.goo = goo
        self.threshold = threshold

    def _distance(
        self,
        lat1: NDArray[np.float64],
        lon1: NDArray[np.float64],
        lat2: Union[float, NDArray[np.float64]],
        lon2: Union[float, NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        """Euclidean distance in lat/lon degrees, with lon wrapping."""
        # Allow lat2/lon2 to be scalars or arrays; numpy broadcasting will handle operations.
        # Wrap longitudes to [-180, 180]
        dlon = ((lon1 - lon2 + 180) % 360) - 180
        dlat = lat1 - lat2
        return np.sqrt(dlat**2 + dlon**2)

    def field_mask(
        self,
        lats: NDArray[Any],
        lons: NDArray[Any],
    ) -> NDArray[Any]:
        """Compute boolean mask for the metaball field."""
        LAT, LON = np.meshgrid(lats, lons, indexing="ij")  # shape (lat_size, lon_size)
        field = np.zeros_like(LAT, dtype=float)

        for ball in self.balls:
            dist = self._distance(LAT, LON, ball.lat, ball.lon)
            # avoid division by zero
            dist[dist == 0] = 1e-12
            field += ball.size / (dist**self.goo)

        return field >= self.threshold


###################################################################################################
######################################### CLUSTER #################################################
###################################################################################################


class CLUSTER(ShapeMethod):
    def __init__(
        self,
        center_lat: Optional[float] = None,
        center_lon: Optional[float] = None,
        size_lat: Optional[float] = None,
        size_lon: Optional[float] = None,
        n: Optional[int] = None,
        mode: Optional[ShapeMethod] = None,
        shape_seed: Optional[int] = None,
    ) -> None:
        if shape_seed is not None:
            rng = np.random.default_rng(shape_seed)
        else:
            rng = np.random.default_rng()

        if center_lat is None:
            center_lat = 0
        if center_lon is None:
            center_lon = 0
        if size_lat is None:
            size_lat = 180
        if size_lon is None:
            size_lon = 360
        if n is None:
            n = rng.integers(3, 10)

        # Ensure mode is a valid ShapeMethod instance; default to CELL() if not provided.
        if mode is None:
            mode = CELL()

        self.center_lat = center_lat
        self.center_lon = center_lon
        self.size_lat = size_lat
        self.size_lon = size_lon
        self.n = n
        self.mode = mode
        self.shape_seed = shape_seed

        self.param_dict = {
            "shape_type": CLUSTER,
            "center_lat": center_lat,
            "center_lon": center_lon,
            "size_lat": size_lat,
            "size_lon": size_lon,
            "n": n,
            "mode": mode,
            "shape_seed": shape_seed,
        }

    def make_shape(
        self,
        lats: NDArray[np.float64],
        lons: NDArray[np.float64],
    ) -> NDArray[np.bool_]:
        if self.shape_seed is not None:
            rng = np.random.default_rng(self.shape_seed)
        else:
            rng = np.random.default_rng()
            warnings.warn(
                "No random seed set for PONDS Shape Method. Results will not be reproducible."
            )

        lat_min = self.center_lat - self.size_lat / 2
        lat_max = self.center_lat + self.size_lat / 2
        lon_min = self.center_lon - self.size_lon / 2
        lon_max = self.center_lon + self.size_lon / 2
        lon_max = self.center_lon + self.size_lon / 2

        lat_indices = np.where((lats >= lat_min) & (lats <= lat_max))[0]
        lon_indices = np.where((lons >= lon_min) & (lons <= lon_max))[0]

        shape_mask = np.zeros((len(lats), len(lons)), dtype=bool)

        for _ in range(self.n):
            # Ensure indices are Python ints and values are Python floats to avoid ndarray typing issues
            rand_lat_idx = int(rng.choice(lat_indices))
            rand_lon_idx = int(rng.choice(lon_indices))
            center_lat_val = float(lats[rand_lat_idx])
            center_lon_val = float(lons[rand_lon_idx])

            # Work on a deepcopy of the provided mode to avoid mutating a shared instance
            shape_gen = copy.deepcopy(self.mode)

            # Use setattr so static type checkers don't complain about unknown attributes on ShapeMethod
            setattr(shape_gen, "center_lat", center_lat_val)
            setattr(shape_gen, "center_lon", center_lon_val)

            mask_temp = shape_gen.make_shape(lats, lons)
            shape_mask = shape_mask | mask_temp

            # shape_mask[rand_lat_idx, rand_lon_idx] = True

        return shape_mask
