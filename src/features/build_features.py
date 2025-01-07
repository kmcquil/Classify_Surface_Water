#################################################################################
#################################################################################
# Convert raw data into features
#################################################################################
#################################################################################

#################################################################################
# Load modules
#################################################################################
import geopandas as gpd
from shapely.geometry import box
from matplotlib import pyplot as plt
import numpy as np
import rioxarray
import pandas as pd

#################################################################################
# Study area
#################################################################################
# Shapefile from National Hydrography Dataset watershed boundaries HUC10
hucs = gpd.read_file(r"data\raw\WBD_05_HU2_Shape\Shape\WBDHU10.shp")
# Subset to the watershed near Blacksburg, VA
study_area = hucs[hucs["name"] == "Back Creek-New River"]
# Save
study_area.to_file(r"data\processed\study_area.shp")

#################################################################################
# Extract labeled features from the Sentinel-2 dataset
#################################################################################
# Polygons labeled according to land cover created in QGIS using high res google 
# satellite imagery basemap

# Open the S2 image to classify
image = rioxarray.open_rasterio(r"data\processed\s1_monthly_composite_202407.tif")

# Open the study area polygon
study_area = gpd.read_file(r"data\processed\\study_area.shp").to_crs(image.rio.crs)

# Open the class label polygons
training_data = gpd.read_file(r"data\processed\class_labels.shp").to_crs(image.rio.crs)

# Plot the training data
fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))  # figsize goes width, height
study_area.plot(color="grey", legend=False, ax=ax1)
training_data.plot(column="class", cmap=None, legend=True, ax=ax1)
plt.show()

# List land cover classes
training_classes = np.unique(training_data["class"])


# Extract values from polygons
def extract_values(img, poly, label):

    # Subset polygons to the label
    poly = poly[poly["class"] == label]

    # Clip to the polygon and extract values
    img_cropped = img.rio.clip(poly.geometry.values, poly.crs, drop=True).to_numpy()

    # Cleanup
    flat_array = np.transpose(img_cropped.reshape(10, -1))
    df = pd.DataFrame(flat_array)
    df.dropna(inplace=True)
    df.columns = img.long_name
    df["label"] = label
    return df


labeled_df = pd.concat(
    [extract_values(image, training_data, i) for i in training_classes]
)
labeled_df.reset_index(drop=True, inplace=True)
labeled_df.reset_index(drop=False, inplace=True)
labeled_df.to_csv(r"data\processed\labeled_df.csv", index=False)