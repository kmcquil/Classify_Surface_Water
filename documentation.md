# Project Documentation 

Most of my coursework in school taught ML in R. This is a short and simple classification project I set up to become more familiar with using STAC and Scikit-lern. In this project, I classify surface water in the Back Creek New River Watershed from Sentinel-2 imagery using a Scikit-learn Random Forest model.

## Environment 
- Set up a new anaconda env with scikit-learn for this project.


## Data
- Download theh USGS National Hydrography Watershed Boundary Dataset HUC 12 to obtain the boundaries for the Back Creek New River watershed located near Blacksburg, VA
- Generate a Sentinel-2 average cloud-masked composite from July 2024 using the STAC API.
- Data not included in this repo bc it was too big.

## Create a labeled training dataset
- Select 4 simple training labels
    - Water
    - Built-up 
    - Grass/crops
    - Forest
- Visualize the Sentinel-2 RGB image in QGIS and create polygons to extract examples from each class
- Create a df pairing the data from the Sentinel-2 bands with the corresponding land cover label

## Train the model
- For a simple classification task, all Sentinel-2 bands are probably not necessary to achieve good performance. Find which are the smallest combination of bands that yield satisfactory performance, particulary for surface water. 
- Using the subset of bands selected in the previous step, tune the hyper parameters. 

## Predict land cover 
- Use the final model to classify the full image
- Performance metrics on the test dataset
    - overall accuracy = 99%
    - surface water recall = 99%
    - surface water precision = 99%
    - surface water f1 score = 99%
- Performanced changed minimally applying the model to the test dataset instead of the training dataset, indicating it was not overfit. However, this is a really small study area, so that wasn't really a challenge.
- Despite the seemingly nearly perfect performance, I didn't do the best job creating the training labels, so there are certaintly errors in here somewhere.

![Classifcation result](./outputs/true_and_preds.png)
