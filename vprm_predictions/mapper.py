"""compares Copernicus land cover map extent to a MODIS tile extent

main user functions:
 - map_from_disk()
     - plots the full extent of a specified MODIS tile and a Copernicus land
       cover map tile
 - map_before_regrid()
     - plots the extent of the MODIS data and the Copernicus land cover tile
       just before pyVPRM calls ESMF_RegridWeightGen. Demonstrates the
       mis-alignment that (I think) is causing ESMF_RegridWeightGen to crash.
"""

import geopandas as gpd
import hvplot
import hvplot.pandas
import hvplot.xarray
import matplotlib.pyplot as plt
import numpy as np
import nzgeom.coastlines
import pyproj
import rioxarray
import xarray as xr
from loguru import logger
from shapely.geometry import box
import shapely.plotting

tile = "h30v13"
fname_modis = f"/home/timh/mnt/amp/CarbonWatchUrban/urbanVPRM/MODIS/MOD09A1.061_{tile}_20180101_20180322.nc"
fname_lcm = "/home/timh/mnt/raukawa_code/pyVPRM_NewZealand_Modelling/vprm_predictions/data/copernicus/E160S40_PROBAV_LC100_global_v3.0.1_2019-nrt_Discrete-Classification-map_EPSG-4326.tif"
modis_sinu_proj4str = (
    "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m"
    " +no_defs"
)


def calc_bounds_from_centre_1D(x: xr.DataArray | np.ndarray):
    """calculate bounds from centres

    Assumes that the spacing of the input is regular.

    ARGS:
    x: xr.DataArray: 1-D array of centre coordinates. Must be regularly spaced

    RETURNS:
    a numpy array of size (x.size + 1) containing the boundary coordinates
    """
    if isinstance(x, xr.DataArray):
        x = x.values
    d = np.diff(x)
    if not np.allclose(d.min(), d.max()):
        raise (ValueError("input array is not regularly-spaced"))
    d = d.mean()
    x_b = np.linspace(start=x[0] - d, stop=x[-1] + d, num=x.size + 1)
    return x_b


def get_lonlat(ds: xr.Dataset, xcoord: str = "XDim", ycoord: str = "YDim"):
    """add longitude, latitude centre and bound coordinates to a dataset with MODIS sinusoidal coordinates

    Calculates MODIS sinusoidal coordinates of grid bounds from the coordinate
    of grid cell centres. Then calculates longitude and latitude for both
    centres and bounds from the MODIS sinusoidal coordinates. Returns a dataset
    with added coordinates and data variables "longitude", "latitude", "longitude_b", and
    "latitude_b"

    """
    modis_sinu_proj4str = (
        # "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
        "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m"
        " +no_defs"
    )
    proj_mod = pyproj.crs.CRS(projparams=modis_sinu_proj4str)
    proj_latlon = pyproj.crs.CRS(projparams=4326)

    modis_sinu_2_lonlat = pyproj.Transformer.from_crs(
        crs_from=proj_mod, crs_to=proj_latlon
    )
    x, y = np.meshgrid(ds[xcoord].values, ds[ycoord].values, indexing="ij")
    x_b, y_b = np.meshgrid(
        calc_bounds_from_centre_1D(ds[xcoord]),
        calc_bounds_from_centre_1D(ds[ycoord]),
        indexing="ij",
    )
    lat, lon = modis_sinu_2_lonlat.transform(x, y)
    lat_b, lon_b = modis_sinu_2_lonlat.transform(x_b, y_b)

    xcoord_b = xcoord + "_b"
    ycoord_b = ycoord + "_b"

    ds = ds.assign_coords(
        {
            "lon": ((ycoord, xcoord), lon.transpose()),
            "lat": ((ycoord, xcoord), lat.transpose()),
        }
    )
    ds = ds.assign_coords(
        {
            "lon_b": ((ycoord_b, xcoord_b), lon_b.transpose()),
            "lat_b": ((ycoord_b, xcoord_b), lat_b.transpose()),
        }
    )
    return ds


def map_from_disk():
    logger.info("starting!")
    xds_mod_sinu = rioxarray.open_rasterio(fname_modis)
    xds_mod = get_lonlat(xds_mod_sinu, xcoord="x", ycoord="y")
    # logger.info("reprojecting MODIS tile to lon/lat")
    # xds_mod = xds_mod_sinu.rio.write_crs(modis_sinu_proj4str).rio.reproject("EPSG:4326")
    # logger.info("done reprojecting")

    xds_mod = xds_mod.assign_coords(lon360=(xds_mod["lon"] + 360) % 360)
    coasts = nzgeom.coastlines.get_NZ_coastlines()

    xds_lcm = rioxarray.open_rasterio(fname_lcm)
    bounds_lcm = box(*xds_lcm.rio.bounds())

    # # plot with hvplot (slow, zoomable)
    # p_coasts = coasts.hvplot()
    # p_mod = (
    #     xds_mod["sur_refl_b01"]
    #     .isel(time=0)
    #     .drop_attrs()
    #     .hvplot.quadmesh("lon360", "lat", "sur_refl_b01")
    # )
    # hvplot.show(p_mod * p_coasts)

    # plot with matplotlib (fast, not zoomable)
    fig, ax = plt.subplots()
    coasts.plot(ax=ax, color="lightgray", label="New Zealand")
    b01 = xds_mod["sur_refl_b01"].isel(time=0)
    # b01 = b01.where(b01 > b01.attrs['_FillValue'])
    pcm_b01 = ax.pcolormesh(
        (xds_mod["lon"] + 360) % 360,
        xds_mod["lat"],
        b01,
    )
    # pcm_lcm = ax.pcolormesh(
    #     (xds_lcm["x"] + 360) % 360,
    #     xds_lcm["y"],
    #     xds_lcm.values.squeeze(),
    #     cmap='Greys'
    # )
    shapely.plotting.plot_polygon(bounds_lcm, ax=ax)
    plt.colorbar(pcm_b01, ax=ax)
    ax.set_title(f"MODIS tile {tile}")


def map_before_regrid():
    coasts = nzgeom.coastlines.get_NZ_coastlines()
    LONLAT = "EPSG:4326"  # EPSG ID for lon/lat coordinates
    # source (MODIS) to geodataframe
    xds_dst_modis = xr.open_dataset("./grid_dst.nc")
    bounds_dst = box(*xds_dst_modis.rio.bounds())
    gdf_bounds_dst = gpd.GeoDataFrame(
        {"id": "destination MODIS", "geometry": [bounds_dst]},
        crs=modis_sinu_proj4str,
    )
    # gdf_bounds_dst = gdf_bounds_dst.to_crs(LONLAT)
    # destination (copernicus) to geodataframe
    xds_src_copernicus = xr.open_dataset("./grid_src.nc")
    bounds_src = box(*xds_src_copernicus.rio.bounds())
    gdf_bounds_src = gpd.GeoDataFrame(
        {"id": "source Copernicus", "geometry": [bounds_src]}, crs=LONLAT
    )
    # plot with NZ coastlines
    p_coasts = coasts.to_crs(modis_sinu_proj4str).hvplot()
    p_dst = gdf_bounds_dst.hvplot.polygons(hover_cols=["id"]).opts(
        alpha=0.2, color="red"
    )
    p_src = (
        gdf_bounds_src.to_crs(modis_sinu_proj4str)
        .hvplot.polygons(hover_cols=["id"])
        .opts(alpha=0.2, fill_color="lightcoral")
    )
    gdf = pd.concat([gdf_bounds_dst, gdf_bounds_src.to_crs(modis_sinu_proj4str)])
    p_src_dst = gdf.hvplot.polygons(c="id").opts(alpha=0.2)
    p = (p_coasts * p_src_dst).opts(
        height=800, width=1600, fontscale=2.0, title=f"MODIS tile {tile}"
    )
    hvplot.show(p)
