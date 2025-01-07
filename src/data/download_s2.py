#################################################################################
#################################################################################
# Download cloud free composite from Sentinel-2 during July 2024 using STAC API
#################################################################################
#################################################################################

# Load modules
import pystac_client
import geopandas as gpd
import odc.stac
import numpy as np
import xarray as xr

# Define the bounding box for the area of interest
study_area = gpd.read_file(r"data\processed\study_area.shp")
study_area_bounds = study_area.bounds.iloc[0]
bbox = [
    study_area_bounds.minx,
    study_area_bounds.miny,
    study_area_bounds.maxx,
    study_area_bounds.maxy,
]

# URL of the STAC catalog to query
url = "https://earth-search.aws.element84.com/v1"

# Initialize a STAC client to interact with the specified STAC catalog
catalog = pystac_client.Client.open(url)

# Build the query
start_date = "2024-07-01"
end_date = "2024-07-31"
collections = ["sentinel-2-l2a"]
query = catalog.search(
    bbox=bbox,
    collections=collections,
    datetime=f"{start_date}/{end_date}",
    query=["eo:cloud_cover<20"],
)

# Fetch the items
items = list(query.items())

# Define Coordinate Reference System (CRS) to which all the data should be reprojected
crs_stac = "EPSG:32617"

# Define the pixel resolution of your data
resolution = 10

# Load satellite imagery using the `load` function from odc.stac. This function retrieves the specified bands and organizes them into an xarray dataset.
ds = odc.stac.load(
    items,
    crs=crs_stac,
    resolution=resolution,
    chunks={},
    groupby="solar_day",
    bbox=bbox,
)
# Cloud mask using the scene classification band
valid = (
    (ds.scl == 2)  # dark area pixels
    | ((ds.scl >= 4) & (ds.scl <= 7))  # vegetation, not vegetated, water, unclassified
    | (ds.scl == 11)  # snow/ice
)
ds_masked = xr.where(valid, ds, np.nan)

# Calculate the mean
ds_masked_avg = ds_masked.mean(dim="time")

# Calculate NDVI (NIR-Red)/(NIR+Red)
ds_masked_avg["ndvi"] = (ds_masked_avg["nir"] - ds_masked_avg["red"]) / (
    ds_masked_avg["nir"] + ds_masked_avg["red"]
)

# Calculate NDWI  (G-NIR)/(G+NIR)
ds_masked_avg["ndwi"] = (ds_masked_avg["green"] - ds_masked_avg["nir"]) / (
    ds_masked_avg["green"] + ds_masked_avg["nir"]
)

# Cast all bands to same type
ds_masked_avg = ds_masked_avg.astype("float64")

# Select a subset of the bands since there are so many
ds_masked_avg = ds_masked_avg[
    [
        "blue",
        "green",
        "red",
        "nir",
        "rededge1",
        "scl",
        "swir16",
        "ndwi",
        "ndvi",
        "visual",
    ]
]

# Fix the crs
ds_masked_avg.rio.write_crs(crs_stac, inplace=True).rio.set_spatial_dims(
    x_dim="x", y_dim="y", inplace=True
).rio.write_coordinate_system(inplace=True)

# Save image
ds_masked_avg.rio.to_raster(r"data\processed\s1_monthly_composite_202407.tif")
