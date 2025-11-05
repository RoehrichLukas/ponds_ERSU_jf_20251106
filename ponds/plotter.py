import matplotlib.pyplot as plt
import cartopy.crs as ccrs  # type: ignore
import cartopy.feature as cfeature  # type: ignore
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from typing import Any
from numpy.typing import NDArray

# from build.lib.build.lib.ponds.core import PONDS
# import ponds


def worldmap(
    ponds: Any,
    verbose: bool = False,
    figsize: tuple[float, float] = (16, 8),
    coastline: bool = True,
    borders: bool = True,
    show_cells: bool = False,
    shift_type: bool = False,
    title: str = "Cluster Masks",
) -> tuple[Figure, Axes]:
    """Plot the PONDS data on a world map using lat/lon as axis ticks.

    Args:
        ponds: PONDS dataset containing cluster masks.
        verbose: If True, print additional information during processing.
        figsize: Size of the figure to create.
        coastline: If True, add coastline features to the map.
        borders: If True, add country borders to the map.
        show_cells: If True, draw grid lines based on xarray resolution.
        title: Title for the plot.
    """

    # Extract cluster masks
    cluster_masks: list[NDArray[np.float64]] = []
    if verbose:
        print("Extracting cluster masks...")
    cluster_key = ponds.data.attrs["cluster_key"]
    for name, da in ponds.data.data_vars.items():
        if name.startswith(cluster_key):
            if verbose:
                print(f"Found cluster variable: {name}")
            cluster_masks.append(da)
    if verbose:
        print(f"Found {len(cluster_masks)} cluster variables.")

    # Specify Colors
    cland = "#fff3e0"
    cocean = "#e0f7fa"
    cblobs = [
        "#d36060",  # soft red
        "#f0a830",  # warm orange
        "#f6d55c",  # pastel yellow
        "#7bc043",  # fresh green
        "#69d2e7",  # sky blue
        "#6384b3",  # dusty blue
        "#f17cb0",  # pink rose
        "#c894d5",  # light purple
        "#ff9671",  # coral
        "#52b788",  # soft teal
        "#6c5ce7",  # indigo
    ]

    # Set up map with Cartopy
    fig, ax = plt.subplots(  # type: ignore
        figsize=figsize, subplot_kw={"projection": ccrs.PlateCarree()}
    )

    # Add map features
    if borders:
        ax.add_feature(cfeature.BORDERS, linestyle=":")  # type: ignore
        ax.add_feature(cfeature.LAND, facecolor=cland)  # type: ignore
        ax.add_feature(cfeature.OCEAN, facecolor=cocean)  # type: ignore
    if coastline:
        ax.add_feature(cfeature.COASTLINE)  # type: ignore
        ax.add_feature(cfeature.LAND, facecolor=cland)  # type: ignore
        ax.add_feature(cfeature.OCEAN, facecolor=cocean)  # type: ignore
    if not show_cells:
        ax.gridlines(  # type: ignore
            draw_labels=True,
            colors="gray",
            linestyle="-.",
        )
    else:
        ax.gridlines(draw_labels=True, color="none")  # type: ignore

    for i, mask in enumerate(cluster_masks):
        collapsed = mask.any(dim="time").astype(int)  # type: ignore

        # Get lat/lon coordinates safely
        lat = collapsed.coords["lat"].values  # type: ignore
        lon = collapsed.coords["lon"].values  # type: ignore

        # Define image extent from lat/lon
        extent: list[float] = [lon.min(), lon.max(), lat.min(), lat.max()]  # type: ignore

        # Plot the mask on the map
        ax.imshow(  # type: ignore
            collapsed,  # type: ignore
            cmap=ListedColormap(["none", cblobs[i % len(cblobs)]]),
            alpha=0.6,
            extent=extent,  # type: ignore
            origin="lower",
            transform=ccrs.PlateCarree(),
        )

        # Add contour outlines (white)
        ax.contour(  # type: ignore
            lon,  # type: ignore
            lat,  # type: ignore
            collapsed.astype(int),  # type: ignore
            levels=[0.5],
            colors=cblobs[i % len(cblobs)],
            linewidths=2,
            transform=ccrs.PlateCarree(),
        )

    # Add optional grid lines based on lat/lon spacing
    if show_cells:
        lat = ponds.data.coords["lat"].values  # type: ignore
        lon = ponds.data.coords["lon"].values  # type: ignore

        if verbose:
            print("Drawing grid lines based on xarray resolution...")
        # Compute midpoints and deltas
        lat_edges = _get_grid_edges(lat)  # type: ignore
        lon_edges = _get_grid_edges(lon)  # type: ignore

        for x in lon_edges:
            ax.plot(  # type: ignore
                [x, x],
                [lat_edges[0], lat_edges[-1]],
                color="black",
                linewidth=0.5,
                alpha=0.3,
                transform=ccrs.PlateCarree(),
                zorder=10,
            )

        for y in lat_edges:
            ax.plot(  # type: ignore
                [lon_edges[0], lon_edges[-1]],
                [y, y],
                color="black",
                linewidth=0.5,
                alpha=0.3,
                transform=ccrs.PlateCarree(),
                zorder=10,
            )

    # Add optional shift type in the center of the cluster
    if shift_type:
        if verbose:
            print("Adding shift type labels...")
        for i, mask in enumerate(cluster_masks):
            # Get the center of the mask
            center_lat = mask.attrs["center_lat"]  # type: ignore
            center_lon = mask.attrs["center_lon"]  # type: ignore
            shift_type = str(mask.attrs["shift_type"]).split(".")[-1][:-2]  # type: ignore
            shape_type = str(mask.attrs["shape_type"]).split(".")[-1][:-2]  # type: ignore
            if verbose:
                print(f"Shift type for mask {i + 1}: {shift_type}")

            if shape_type == "CELL":
                pos_lon = center_lon
                pos_lat = center_lat + 5
            else:
                pos_lon = center_lon
                pos_lat = center_lat
            # Add text label at the center with a non-transparent box
            ax.text(  # type: ignore
                pos_lon,  # type: ignore
                pos_lat,  # type: ignore
                f"{shift_type}",
                fontsize=10,
                color="black",
                ha="center",
                va="center",
                transform=ccrs.PlateCarree(),
                bbox=dict(
                    facecolor=cblobs[i % len(cblobs)],
                    edgecolor="none",
                    alpha=1.0,
                    boxstyle="round,pad=0.3",
                ),
            )

    ax.set_title(title)  # type: ignore

    return fig, ax


def _get_grid_edges(
    coords: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Helper to calculate grid edges from 1D coordinate centers."""

    # Handle trivial case of a single coordinate
    if coords.size < 2:
        return coords.copy()

    # Calculate edge positions
    step = (coords[-1] - coords[0]) / (len(coords))
    edges: NDArray[np.float64] = np.arange(
        coords[0], coords[-1] + step, step, dtype=np.float64
    )
    return edges


def plot_all_ts(
    ponds: Any,
    alpha: float = 0.5,
    linewidth: float = 0.5,
    figsize: tuple[float, float] = (16, 4),
    title: str = "Time Series for All Cells",
) -> tuple[Figure, Axes]:
    """
    Plots the time series of every grid cell from the PONDS dataset.
    """

    fig, ax = plt.subplots(  # type: ignore
        figsize=figsize,
    )

    for i in range(ponds.data.sizes["lat"]):
        for j in range(ponds.data.sizes["lon"]):
            ax.plot(  # type: ignore
                ponds.data.ts[:, i, j],
                linestyle="-",
                linewidth=linewidth,
                marker=None,
                color="grey",
                alpha=alpha,
            )

    # eyecandy
    ax.set_title(title)  # type: ignore
    ax.set_xlabel("Time")  # type: ignore
    ax.set_ylabel("Value")  # type: ignore

    return fig, ax
