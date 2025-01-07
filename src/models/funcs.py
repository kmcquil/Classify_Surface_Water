#################################################################################
# Functions 
#################################################################################

# Load modules
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import time


def train_model(
    att_list, train_features, train_labels, validation_features, validation_labels
):
    """
    Train the random forest model with a subset of features, assess performance on validation set,
      and return pd df of performance
    att_list: list of attributes to include in the model
    train_features: pd df of training dataset
    train_labels: label for each feature in training dataset
    validation_features: pd df of validation dataset
    validation_labels: label for each feature in validation dataset
    """
    # Subset to the attributes of interest
    train_features = train_features[att_list]
    validation_features = validation_features[att_list]

    # Instantiate model with 1000 trees
    rf = RandomForestClassifier(n_estimators=1000, random_state=42)

    # Train the model on training data
    start_time = time.time()
    rf.fit(train_features, train_labels)
    end_time = time.time()
    train_time = end_time - start_time

    # Predict on test data
    validation_preds = rf.predict(validation_features)

    # Assess model performance, paying specific attention to surface water
    # precision: Of all the times the model predicted positive, how many were actually positive?
    # recall: Of all the actual positive instances, how many did the model correctly identify?
    # f1: This metric is the harmonic mean of precision and recall, providing a balanced measure of both metrics.
    # It's particularly useful when dealing with imbalanced datasets, where one class is significantly more prevalent than the other.
    classification_report = metrics.classification_report(
        validation_labels, validation_preds, output_dict=True
    )
    results = {
        "accuracy_overall": metrics.accuracy_score(validation_labels, validation_preds),
        "precision_water": classification_report["water"]["precision"],
        "recall_water": classification_report["water"]["recall"],
        "f1_water": classification_report["water"]["f1-score"],
        "train_time": train_time,
        "n_feats": len(att_list),
        "feats": ",".join(att_list),
    }
    return results
