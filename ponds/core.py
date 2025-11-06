import numpy as np
import xarray as xr
import warnings

from typing import Optional, Any
from numpy.typing import NDArray

from ponds import (
    plotter,
)
from ponds.shapegen import ShapeMethod, SINGLE
from ponds.shiftgen import ShiftMethod
from ponds.backgroundgen import BackgroundMethod
from ponds.correlationgen import CorrelationMethod, NoCorr

# import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


class PONDS:
    """
    Main class for PONDS dataset generation.

    Represents the dataset containing of spatial grid cells. Each grid cell contains a time series
    of base values (e.g. background noise). In certain areas additional features (e.g. abrupt shifts)
    are clustered.

    A shift can be added by calling the class function add_shift with custom shape- and shift-generation
    methods. Each shift is stored as a boolean mask in the underlying xarray.Dataset and labeled using
    the cluster_key argument.

    The time series data itself is stored in the `ts` DataArray.

    Args:
        lat_size (int): Number of latitude points
        lon_size (int): Number of longitude points
        time_size (int): Number of time points, governs temporal resolution
        background_trend (float): Linear trend coefficient
        background_noise (float): Amplitude of random noise
        n_shifts (int): Number of abrupt shift events to add
        random_seed (int): Seed for reproducible results
        cluster_key (str): Key to use for the variable name in the time series parameter grid
                           (e.g. "c" for "c1", "c2", ...).
    """

    def __init__(
        self,
        lat_size: int = 1,
        lon_size: int = 1,
        time_size: int = 100,
        cluster_key: str = "c",
        background_key: str = "b",
    ) -> None:
        """
        Generate spatial grid and seed with time series generator parameters.
        """

        # 1) Set up coordinates
        self.lats = np.linspace(-90, 90, lat_size)
        self.lons = np.linspace(-180, 180, lon_size)
        self.time = np.linspace(0, time_size - 1, time_size)

        # 2) Generate background data with trend and noise
        data_ts = np.zeros((time_size, lat_size, lon_size))

        # - Create xarray DataArray, init with background time series data
        dims = ("time", "lat", "lon")
        da_ts = xr.DataArray(
            data_ts,
            dims=dims,
            coords={
                "time": self.time,
                "lat": self.lats,
                "lon": self.lons,
            },
        )
        # - Create xarray Dataset to hold the time series data
        self.data = xr.Dataset(
            {
                "ts": da_ts,
            },
            coords={"time": self.time, "lat": self.lats, "lon": self.lons},
            attrs={
                "cluster_key": cluster_key,
                "background_key": background_key,
            },
        )

    def add_background(
        self,
        background_method: BackgroundMethod,
        shape_method: ShapeMethod = SINGLE(),
    ) -> None:
        """
        Adds a background to the time series data.

        Background Methods:
            - "WhiteNoise": Adds white noise background to the time series.
            - "Trend": Adds a linear trend to the time series.

        Args:
            background_method (BackgroundMethod): Method to generate the background.
        """

        # 1) Generate shape mask if shape_method is provided
        # - warn users to make sure that shape is handled correctly
        if shape_method is SINGLE():
            if self.data.sizes["lat"] != 1 or self.data.sizes["lon"] != 1:
                warnings.warn(
                    "No shape method provided for a non-single ponds grid. "
                    "Shift method is applied to all cells."
                )
        else:
            if self.data.sizes["lat"] == 1 and self.data.sizes["lon"] == 1:
                warnings.warn(
                    "Shape method specified for a (1x1) grid. PONDS is in single-mode. "
                    "If chosen shape method is not SINGLE it is likely to fail."
                )

        # - generate shape mask
        shape_mask = shape_method.make_shape(
            lats=self.lats,
            lons=self.lons,
        )
        shape_dict = shape_method.param_dict

        # check for seed for random process
        if hasattr(background_method, "background_seed"):
            if getattr(background_method, "background_seed", None) is not None:
                rng = np.random.default_rng(
                    getattr(background_method, "background_seed")
                )
            else:
                rng = np.random.default_rng()
                warnings.warn(
                    "No random seed set for PONDS Background Method. Results will not be reproducible."
                )
            background_method.rng = rng

        # 2) Generate background data and add to time series
        for i in range(self.data.sizes["lat"]):
            for j in range(self.data.sizes["lon"]):
                if shape_mask[i, j]:
                    background_effect = background_method.make_background(
                        time_size=len(self.time),
                    )

                    # add shift effect to the data in single grid cell, all time steps at once
                    self.data.ts.values[:, i, j] += background_effect

        # 3) store background metadata
        self._add_background_to_ds(
            background_dict=background_method.param_dict,
            shape_dict=shape_dict,
        )

    def add_shift(
        self,
        shift_method: ShiftMethod,
        shape_method: ShapeMethod = SINGLE(),
        correlation_method: CorrelationMethod = NoCorr(),
    ) -> None:
        """
        Adds a shift to the time series data.

        Shape Methods:
            - "ELLIPSIS": Adds a shift in the shape of an eliptic blob.
            - "BLOB": Adds a shift in the shape of a random blob.
            - "CELL": Adds a shift in a single grid cell.
            - "CLUSTER": Adds several cluster shapes at various locations.
            - "CUSTOM": Adds a shift in a user-defined shape.
            - ...
        Shift Methods:
            - "SIGMOID": Adds a shift using a sigmoid function.
            - "BIMODAL": Adds a shift switching between two levels.
            - ...
        Correlation Methods:
            - "Noise": Adds random time offsets to the shift times.
            - "Gauss": Adds time offsets based on a 2D Gaussian distribution.
            - None: No time offsets, all shifts happen at the same time.
            - ...

        Args:
            shape_method (ShapeMethod): Method to generate the shape of the shift.
            shift_method (ShiftMethod): Method to generate the shift in the time series.
            random_seed (int, optional): Seed for reproducibility. Defaults to None.
            kwargs (dict, optional): Additional keyword arguments for the shift method.
        """

        # 0) preprocessing
        # - align cluster shape with cluster correlation if necessary
        self._align_cluster_shape_with_cluster_correlation(
            shape_method, correlation_method
        )

        # - warn users to be aware of SINGLE-mode usage
        if shape_method is SINGLE():
            if self.data.sizes["lat"] != 1 or self.data.sizes["lon"] != 1:
                warnings.warn(
                    "No shape method provided for a non-single ponds grid. "
                    "Shift method is applied to all cells."
                )
        else:
            if self.data.sizes["lat"] == 1 and self.data.sizes["lon"] == 1:
                warnings.warn(
                    "Shape method specified for a (1x1) grid. PONDS is in single-mode. "
                    "If chosen shape method is not SINGLE it is likely to fail."
                )

        # - check if shift_method param_dict contains 'shift_type' key
        if "shift_type" not in shift_method.param_dict:
            warnings.warn(
                "Shift method param_dict does not contain 'shift_type' key. "
                "Setting shift_type to 'UNKNOWN'."
            )
            shift_method.param_dict["shift_type"] = "UNKNOWN"
        # - check if shape_method param_dict contains 'shape_type' key
        if "shape_type" not in shape_method.param_dict:
            warnings.warn(
                "Shape method param_dict does not contain 'shape_type' key. "
                "Setting shape_type to 'UNKNOWN'."
            )
            shape_method.param_dict["shape_type"] = "UNKNOWN"

        # 1) generate shape mask
        # - make the boolean shape mask
        shape_mask = shape_method.make_shape(
            lats=self.lats,
            lons=self.lons,
        )
        # - check if shape_mask is valid -> raise error if no True values in it
        if not shape_mask.any():
            raise ValueError("Shape mask is empty. Please check the shape method.")

        # 2) generate time offsets according to correlation method
        time_offsets = correlation_method.make_time_offsets(
            shape_mask
        )  # -> generate offset values

        # 3) generate shift data
        mask3d = np.zeros(
            (self.data.sizes["time"], *shape_mask.shape), dtype=bool
        )  # provide some space -> will be later added as cluster DataArray to the Dataset
        # - iterate through every cell
        for i in range(self.data.sizes["lat"]):
            for j in range(self.data.sizes["lon"]):
                if shape_mask[i, j]:
                    # - generate shift data using the shift method
                    shift_effect, t_shift = shift_method.make_shift(
                        time_arr=self.time, time_offset=time_offsets[i, j]
                    )

                    # - add shift effect to the data in single grid cell, all time steps at once
                    self.data.ts.values[:, i, j] += shift_effect
                    # - update mask3d to indicate where the shift was applied
                    mask3d[t_shift, i, j] = True

        # 4) documentation of shift cluster event
        # - store shift as a DataArray to the Dataset -> stores all the information to identify and reproduce the shift event
        self._add_shift_to_ds(
            mask=mask3d,
            shift_dict=shift_method.param_dict,
            shape_dict=shape_method.param_dict,
            correlation_dict=correlation_method.param_dict,
        )

    def _get_new_background_key(self) -> str:
        """
        Generate a new background key based on the current number of backgrounds.
        """

        # get all the attribute keys in the ts DataArray
        variable_names = list(self.data.ts.attrs.keys())
        # filter the keys that start with the background_key
        variable_names = [
            name
            for name in variable_names
            if self.data.ts.attrs[name] == "### HEADER ###"
        ]
        # get the last variable index
        last_variable_index = int(len(variable_names))
        # return the new background key
        return f"{self.data.attrs['background_key']}{last_variable_index + 1}"

    def _get_new_cluster_key(self) -> str:
        """
        Generate a new cluster key based on the current number of shifts.
        The cluster key is used to name the variables in the Dataset.
        """
        variable_names = list(self.data.data_vars.keys())
        variable_names = [
            name
            for name in variable_names
            if name.startswith(self.data.attrs["cluster_key"])
        ]
        last_variable_index = int(len(variable_names))
        return f"{self.data.attrs['cluster_key']}{last_variable_index + 1}"

    def _add_background_to_ds(
        self,
        background_dict: dict[str, Any],
        shape_dict: dict[str, Any],
    ) -> None:
        """
        Appends metadata to the attributes of the time series DataArray.

        Args:
            background_dict (dict): Dictionary containing parameters used to generate the background.
            shape_dict (dict): Dictionary containing parameters used to generate the shape of the background.
        """
        # get variable name for new background
        new_background_name = self._get_new_background_key()
        # add background attributes to the ts DataArray
        self.data.ts.attrs[new_background_name] = "### HEADER ###"
        # - add background attributes
        for k, v in background_dict.items():
            self.data.ts.attrs[f"{new_background_name}_{k}"] = v
        # - add shape attributes
        for k, v in shape_dict.items():
            self.data.ts.attrs[f"{new_background_name}_{k}"] = v

    def _add_shift_to_ds(
        self,
        mask: NDArray[np.bool_],
        shift_dict: dict[str, Any],
        shape_dict: dict[str, Any],
        correlation_dict: dict[str, Any],
    ) -> None:
        """
        Append a xr.DataArray to the ponds Dataset in form of a boolean mask. It identifies
        the spatial and temporal location of the shift event generated by a user specified shift method.
        The parameters used to generate the shift are stored as attributes of the DataArray.

        This method assumes that each added cluster is stored in a separate variable in the
        time series parameter grid, where the cluster_key is used as the variable name followed
        by a number (e.g. "c1", "c2", ...).

        Args:
            mask (np.ndarray): Boolean mask indicating where to append the shift ID.
            param_dict (dict): Dictionary containing parameters used to generate the shift.
            cluster_key (str): Key to use for the variable name in the time series parameter grid
                               (e.g. "c" for "c1", "c2", ...).
        """

        # get variable name for new cluster
        new_cluster_name = self._get_new_cluster_key()

        # Create a new DataArray with the same shape as the mask
        new_da = xr.DataArray(mask, dims=self.data.dims, coords=self.data.coords)
        # Add attributes to the DataArray
        # - add shift attributes
        for k, v in shift_dict.items():
            new_da.attrs[k] = v
        # - add shape attributes
        for k, v in shape_dict.items():
            new_da.attrs[k] = v
        # - add correlation attributes
        for k, v in correlation_dict.items():
            new_da.attrs[k] = v

        # Append the new DataArray to the Dataset
        self.data[new_cluster_name] = new_da

    def _align_cluster_shape_with_cluster_correlation(
        self,
        shape_method: ShapeMethod,
        correlation_method: CorrelationMethod,
    ) -> None:
        """
        Checks if the used ShapeMethod and CorrelationMethod both use a central coordinate. If so,
        it further checks if the central coordinate of the CorrelationMethod is specified by the user.
        If not, it sets the central coordinate of the CorrelationMethod to be the same as the one
        of the ShapeMethod.

        Args:
            shape_method (CorrelationMethod): The shape method used for the cluster.
            correlation_method (CorrelationMethod): The correlation method used for the cluster.
        """

        # check latitudes
        if hasattr(correlation_method, "corr_center_lat"):
            if getattr(correlation_method, "corr_center_lat", None) is None:
                if hasattr(shape_method, "center_lat"):
                    val: float = getattr(shape_method, "center_lat")
                    setattr(correlation_method, "corr_center_lat", val)
                    correlation_method.param_dict["corr_center_lat"] = val
                else:
                    low = float(self.lats.min())
                    high = float(self.lats.max())
                    val: float = float(np.random.uniform(low, high))
                    setattr(correlation_method, "corr_center_lat", val)
                    correlation_method.param_dict["corr_center_lat"] = val

        # check longitudes
        if hasattr(correlation_method, "corr_center_lon"):
            if getattr(correlation_method, "corr_center_lon", None) is None:
                if hasattr(shape_method, "center_lon"):
                    val: float = getattr(shape_method, "center_lon")
                    setattr(correlation_method, "corr_center_lon", val)
                    correlation_method.param_dict["corr_center_lon"] = val
                else:
                    low = float(self.lons.min())
                    high = float(self.lons.max())
                    val: float = float(np.random.uniform(low, high))
                    setattr(correlation_method, "corr_center_lon", val)
                    correlation_method.param_dict["corr_center_lon"] = val

    ###################################################################################################
    ######################################### Get Functions ###########################################
    ###################################################################################################

    def get_ts(
        self,
        single: bool = False,
    ) -> NDArray[np.float64]:
        """
        Retrieve all time series data as a numpy array.

        Args:
            single (bool): If True, return a single flattened array of time series data.
                          If False, return a 3D array with dimensions (time, lat, lon).
        Returns:
            np.ndarray: Time series data in the specified format.
        """

        if single:
            return self.data.ts.values.flatten()
        else:
            return self.data.ts.values

    def get_cluster_masks(
        self,
        cluster_key: str = "c",
    ) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
        """
        Returns all the variables that start with 'cluster_key'. These arrays are
        the masks that identify if a shift happened in a given space-time-coordinate.

        Args:
            cluster_key (str):
        """

        masks: list[NDArray[np.bool_]] = []
        shift_ids: list[float] = []
        for name, da in self.data.data_vars.items():
            if name.startswith(cluster_key):
                # get shift_id: cast values to numeric type to ensure a known dtype for comparisons
                da_vals = da.values.astype(np.float64)  # type: ignore
                ids = np.unique(da_vals)
                if len(ids[ids >= 0]) > 1:
                    warnings.warn(
                        "More than one shift-id found in cluster mask."
                        "Smallest shift-id chosen to proceed..."
                    )
                shift_ids.append(ids[ids >= 0][0])  # type: ignore

                # convert shift_id-mask to boolean mask
                da_mask = np.zeros(da.shape, dtype=bool)
                mask_ix = np.argwhere(da_vals >= 0)
                da_mask[mask_ix[:, 0], mask_ix[:, 1], mask_ix[:, 2]] = True
                masks.append(da_mask)  # type: ignore

        return np.array(masks), np.array(shift_ids)

    def get_ponds_dict(self) -> dict[str, Any]:
        """
        Extracts the metadata from the PONDS instance neccessary to initialize
        a PONDS replica and returns it as a dictionary.

        Returns:
            dict: Dictionary containing the metadata of the PONDS instance.
        """
        params: dict[str, Any] = {
            "lat_size": self.data.sizes["lat"],
            "lon_size": self.data.sizes["lon"],
            "time_size": self.data.sizes["time"],
            "cluster_key": self.data.attrs["cluster_key"],
        }
        return params

    def get_background_instance(
        self, background_key: str
    ) -> tuple[BackgroundMethod, ShapeMethod]:
        """
        Extracts the background parameters from the ponds object ts DataArray.
        Args:
            ponds (PONDS): The ponds object containing the background data.
            background_key (str): The key for the background in the ponds data.

        Returns:
            instance1: The background instance with its parameters.
            instance2: The shape instance with its parameters.
        """
        # get list of keys in order of appearance
        keys = list(self.data.ts.attrs.keys())
        # filter keys that start with the background_key
        # keys = [key for key in keys if key.startswith(background_key)]
        # get index of "background_type" and "shape_type"
        b_index = keys.index(background_key + "_background_type")
        s_index = keys.index(background_key + "_shape_type")

        b_specs: dict[str, Any] = {
            "class": self.data.ts.attrs[keys[b_index]],
            "params": {},
        }
        for key in keys[b_index + 1 : s_index]:
            # remove prefix from key
            key_parts = key.split(background_key + "_")
            # save the parameters in the dictionary
            b_specs["params"][key_parts[1]] = self.data.ts.attrs[key]
        b_instance = b_specs["class"](**b_specs["params"])

        s_specs: dict[str, Any] = {
            "class": self.data.ts.attrs[keys[s_index]],
            "params": {},
        }
        for key in keys[s_index + 1 :]:
            # remove prefix from key
            key_parts = key.split(background_key + "_")
            # save the parameters in the dictionary
            try:
                s_specs["params"][key_parts[1]] = self.data.ts.attrs[key]
            except AttributeError:
                print(
                    f"Warning: Key 'params' not found in shape attributes of {background_key}."
                )

        s_instance = s_specs["class"](**s_specs["params"])

        return b_instance, s_instance

    def get_cluster_instance(
        self, cluster: str
    ) -> tuple[ShiftMethod, ShapeMethod, CorrelationMethod]:
        """
        Extracts the shift, shape and correlation metadata from a given cluster mask.

        Args:
            ponds (PONDS): The PONDS instance to extract metadata from.
            cluster (str): The cluster to filter the metadata attributes.

        Returns:
            dict: A dictionary containing the shift metadata.
        """

        # get list of keys in order of appearance
        keys = list(self.data[cluster].attrs.keys())
        # find index of "shape_type" and "correlation_type"
        shape_type_index = keys.index("shape_type")
        correlation_type_index = keys.index("correlation_type")

        # split keys into shift, shape and correlation dicts
        # - shift dict contains all keys before "shape_type"
        shift_dict: dict[str, Any] = {
            "class": self.data[cluster].attrs["shift_type"],
            "params": {},
        }
        for key in keys[1:shape_type_index]:
            shift_dict["params"][key] = self.data[cluster].attrs[key]
        # - create shift instance
        shift_instance = shift_dict["class"](**shift_dict["params"])

        # - shape dict contains all keys between "shape_type" and "correlation_type"
        shape_dict: dict[str, Any] = {
            "class": self.data[cluster].attrs["shape_type"],
            "params": {},
        }
        for key in keys[shape_type_index + 1 : correlation_type_index]:
            shape_dict["params"][key] = self.data[cluster].attrs[key]
        # - create shape instance
        shape_instance = shape_dict["class"](**shape_dict["params"])

        # - correlation dict contains all keys after "correlation_type"
        correlation_dict: dict[str, Any] = {
            "class": self.data[cluster].attrs["correlation_type"],
            "params": {},
        }
        for key in keys[correlation_type_index + 1 :]:
            correlation_dict["params"][key] = self.data[cluster].attrs[key]
        # - create correlation instance
        correlation_instance = correlation_dict["class"](**correlation_dict["params"])

        return shift_instance, shape_instance, correlation_instance

    ###################################################################################################
    ###################################### Read and Write #############################################
    ###################################################################################################

    def save(
        self,
        filename: str = "ponds_data.nc",
    ) -> None:
        """
        Save the ponds data to a NetCDF file.

        Parameters:
            filename (str): The name of the file to save the data to.
        """
        # make a copy of the data which can be altered for saving
        data_copy = self.data.copy(deep=True)
        # locate all the attributes in every variable which key contains "shape_type" or "background_type" or "shift_type"
        for var in data_copy.data_vars:
            keys = [
                key
                for key in data_copy[var].attrs.keys()
                if "shape_type" in key
                or "background_type" in key
                or "shift_type" in key
                or "correlation_type" in key
            ]
            # for these keys, set the value to string type (netCDF4 does not support class instances)
            for key in keys:
                data_copy[var].attrs[key] = str(data_copy[var].attrs[key])

        # locate all the None entries and replace them with "None"
        for var in data_copy.data_vars:
            for key in data_copy[var].attrs.keys():
                if data_copy[var].attrs[key] is None:
                    data_copy[var].attrs[key] = "None"

        data_copy.to_netcdf(  # type: ignore
            filename,
            engine="netcdf4",
        )

    @staticmethod
    def load(
        filename: str = "ponds_data.nc",
        type_dict: Optional[dict[str, Any]] = None,
    ) -> "PONDS":
        """
        Read the ponds data from a NetCDF file.

        Returns:
            pd (PONDS): An instance of the PONDS class with the data loaded.
        """
        if type_dict is None:
            # if type_dict is not set, use the default type_dict from ponds/__init__.py
            from ponds import type_dict

        with xr.open_dataset(filename, engine="netcdf4") as ds:  # type: ignore
            data = ds.load()  # type: ignore # Detach from file completely

        # locate all the attributes in every variable which key contains "shape_type" or "background_type" or "shift_type"
        for var in data.data_vars:
            keys = [
                key
                for key in data[var].attrs.keys()
                if "shape_type" in key
                or "background_type" in key
                or "shift_type" in key
                or "correlation_type" in key
            ]
            # for these keys, set the value to the class instance using the type_dict stored in ponds/__init__.py
            for key in keys:
                type_str = data[var].attrs[key]
                data[var].attrs[key] = type_dict[type_str.split(".")[-1][:-2]]

        # locate all the 'None' entries and replace them with None
        for var in data.data_vars:
            for key in data[var].attrs.keys():
                if data[var].attrs[key] == "None":
                    data[var].attrs[key] = None

        pd = PONDS.__new__(PONDS)
        pd.data = data
        pd.lats = data.coords["lat"].values  # type: ignore
        pd.lons = data.coords["lon"].values  # type: ignore
        pd.time = data.coords["time"].values  # type: ignore
        return pd

    ###################################################################################################
    #################################### Plotting Functions ###########################################
    ###################################################################################################

    def worldmap(
        self,
        verbose: bool = False,
        figsize: tuple[float, float] = (16, 8),
        borders: bool = True,
        coastline: bool = True,
        show_cells: bool = False,
        shift_type: bool = False,
        title: str = "Cluster Masks",
    ) -> tuple[Figure, Axes]:
        """
        Plot the cluster masks on a world map using Cartopy.

        Args:
            verbose (bool): If True, print additional information during processing.
            figsize (tuple): Size of the figure.
            borders (bool): If True, draw country borders.
            coastline (bool): If True, draw coastlines.
            show_cells (bool): If True, draw grid lines based on xarray resolution.
            title (str): Title of the plot.
        """

        fig, ax = plotter.worldmap(
            ponds=self,
            verbose=verbose,
            figsize=figsize,
            borders=borders,
            coastline=coastline,
            show_cells=show_cells,
            shift_type=shift_type,
            title=title,
        )

        return fig, ax

    def plot_all_ts(
        self,
        linewidth: float = 0.5,
        alpha: float = 0.5,
        figsize: tuple[float, float] = (16, 2),
        title: str = "Time Series for All Cells",
    ) -> tuple[Figure, Axes]:
        """
        Plots the time series of every grid cell from the PONDS dataset.
        """

        fig, ax = plotter.plot_all_ts(
            ponds=self,
            linewidth=linewidth,
            alpha=alpha,
            figsize=figsize,
            title=title,
        )

        return fig, ax
    
    def plot_some_ts(
        self,
        ratio: float = 0.1,
        alpha: float = 0.5,
        linewidth: float = 0.5,
        figsize: tuple[float, float] = (16, 4),
        title: str = "Time Series for Some Example Cells",
    ) -> tuple[Figure, Axes]:
        """
        Plots the time series of selected grid cells from the PONDS dataset.

        Args:
            lat_indices (list[int]): List of latitude indices to plot.
            lon_indices (list[int]): List of longitude indices to plot.
            linewidth (float): Line width for the plots.
            alpha (float): Transparency for the plots.
            figsize (tuple): Size of the figure.
            title (str): Title of the plot.
        """

        fig, ax = plotter.plot_some_ts(
            ponds=self,
            ratio=ratio,
            alpha=alpha,
            linewidth=linewidth,
            figsize=figsize,
            title=title,
        )

        return fig, ax
    
    def plot_single_ts(
            self,
            lat: float,
            lon: float,alpha: float = 0.5,
            linewidth: float = 0.5,
            figsize: tuple[float, float] = (16, 4),
            title: str = "Time Series for Single Cell",
        ) -> tuple[Figure, Axes]:
        """
        Plots the time series of a single grid cell from the PONDS dataset.

        Args:
            lat (float): Latitude of the grid cell to plot.
            lon (float): Longitude of the grid cell to plot.
            linewidth (float): Line width for the plot.
            alpha (float): Transparency for the plot.
            figsize (tuple): Size of the figure.
            title (str): Title of the plot.
        """
        fig, ax = plotter.plot_single_ts(
            ponds=self,
            lat=lat,
            lon=lon,
            alpha=alpha,
            linewidth=linewidth,
            figsize=figsize,
            title=title,
        )

        return fig, ax