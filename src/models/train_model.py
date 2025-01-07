#################################################################################
#################################################################################
# Train random forest model to predict land cover classification
#################################################################################
#################################################################################

#################################################################################
# Load modules
#################################################################################

import os

print(os.getcwd())
import pandas as pd
import numpy as np
import itertools
import sys

sys.path.append("src/models/")
import funcs
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import multiprocessing
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns
import rioxarray
import xarray
from joblib import dump
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.colors as mcolors

#################################################################################
# Load and prep datasets for classification
#################################################################################

# Open the labeled dataset
label_df = pd.read_csv(r"data\processed\labeled_df.csv")

# Convert data to arrays of labels and features
labels = np.array(label_df["label"])
features = label_df.drop(columns=["label", "index", "scl"])
feature_list = list(features.columns)

# Create the training (train + validation) and test split
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, stratify=labels, test_size=0.25, random_state=42
)

# Separate train features into train and validation
train_features, validation_features, train_labels, validation_labels = train_test_split(
    train_features, train_labels, stratify=train_labels, test_size=0.25, random_state=42
)

#################################################################################
# Test all band combinations to see which performs best in the least amount of time
#################################################################################

# List all possible attribute combinations
combs = []
for i in range(1, len(feature_list) + 1):
    els = [list(x) for x in itertools.combinations(feature_list, i)]
    combs.extend(els)

# Use multiprocessing to test all feature combinations
if __name__ == "__main__":
    partial_function = partial(
        funcs.train_model,
        train_features=train_features.copy(),
        train_labels=train_labels.copy(),
        validation_features=validation_features.copy(),
        validation_labels=validation_labels.copy(),
    )
    with multiprocessing.Pool(processes=8) as pool:
        results = pool.map(partial_function, combs)
results = pd.DataFrame(results)

# Plot the relationship between the number of features and performance metrics
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 6))
sns.lineplot(ax=ax1, data=results, x="n_feats", y="accuracy_overall")
sns.lineplot(ax=ax2, data=results, x="n_feats", y="precision_water")
sns.lineplot(ax=ax3, data=results, x="n_feats", y="recall_water")
sns.lineplot(ax=ax4, data=results, x="n_feats", y="f1_water")
plt.show()

# Figure shows that we can achieve just about the same performance using three or four variables as nine variables
# Find the set of features that maximize the accuracy but minimize number of features
# For each number of predictors, get the row with the max accuracy, precision, recall, and f1
perf_summary = results.groupby("n_feats").apply(
    lambda x: x.sort_values(
        ["accuracy_overall", "f1_water", "precision_water", "recall_water"],
        ascending=[False, False, False, False],
    ).head(1)
)

# We decide on using red, rededge1, and swir bc it maxmized performance and minimized training time and n of feats
final_atts = ["red", "rededge1", "swir16"]

#################################################################################
# Tune the hyperparameters
#################################################################################
# Create parameter grid
param_grid = {
    "max_depth": [25, 50, 75, 100],
    "max_features": [2, 3],
    "min_samples_leaf": [3, 4, 5],
    "min_samples_split": [8, 10, 12],
    "n_estimators": [250, 500, 750, 1000],
}
# Use 3-fold cross validation with the grid to identify best set of parameters
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2
)
grid_search.fit(train_features[final_atts], train_labels)
best_grid = grid_search.best_estimator_


#################################################################################
# Assess performance on the test dataset
#################################################################################
print(final_atts)
print(grid_search.best_params_)
test_preds = best_grid.predict(test_features[final_atts])
classification_report = metrics.classification_report(
    test_labels, test_preds, output_dict=True
)
print(metrics.accuracy_score(test_labels, test_preds))
print(classification_report["water"]["precision"])
print(classification_report["water"]["recall"])
print(classification_report["water"]["f1-score"])
# There was no decline in performance - no overfitting!

#################################################################################
# Save the model
#################################################################################

# Save the model
dump(best_grid, r"models\model.joblib")

#################################################################################
# Apply model back to original dataset
#################################################################################

# Load the image and subset to the bands used in classification
image = rioxarray.open_rasterio(r"data\processed\s1_monthly_composite_202407.tif")
image = image.assign_coords(band=list(image.long_name))
image_final_atts = image.loc[final_atts]
image_final_atts.name = "s2_bands"

# Convert to a numpy array
image_array = image_final_atts.to_numpy()
bands, rows, cols = image_array.shape
image_array = np.transpose(image_array.reshape(bands, -1))

# Predict the classification
image_preds = best_grid.predict(image_array)

# Encode the classification
le = LabelEncoder()
le.fit(image_preds)
list(le.classes_)
# 0 = forest, 1 = grass, 2 = urban, 3 = water
image_preds_encoded = le.transform(image_preds)

# Convert back to the original numpy format and merge with the image
image_preds_encoded = image_preds_encoded.reshape(rows, cols)
pred_band = xarray.DataArray(image_preds_encoded, dims=["y", "x"])
pred_band.name = "s2_pred"
merged = xarray.merge([image_final_atts, pred_band])

# Save the predictions
merged["s2_pred"].rio.to_raster(r"outputs\prediction.tif")

#################################################################################
# Viszualize the classification
#################################################################################
image = rioxarray.open_rasterio(r"data\processed\s1_monthly_composite_202407.tif")
image = image[[2,1,0]]
preds = rioxarray.open_rasterio(r"outputs\prediction.tif")

transform = ccrs.epsg(str(image.rio.crs)[-5:])
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(12, 3.5), subplot_kw={"projection": transform}, constrained_layout=True
)
s2_plot = image.plot.imshow(
    ax=ax1,
    transform=transform, 
    robust=True,
    add_labels=False,
)
water_plot = preds.plot(
    ax=ax2,
    transform=transform,
    add_labels=False,
    cmap=mcolors.ListedColormap(["#228B22", "#7CFC00", "#808080", "#0000FF"]),
    add_colorbar=False,
    vmin=0,
    vmax=4,
)
cbar = fig.colorbar(water_plot, ax=ax2, label="", ticks=[0.5, 1.5, 2.5, 3.5])
cbar.set_ticklabels(["Forest", "Grassland/Crops", "Built-up", "Water"])
#plt.show()
plt.savefig(r"outputs\true_and_preds.png" , 
            dpi=700,
            bbox_inches="tight")