def bayesianContentBasedPredict(row, businessVectors, clf, totalBusinesses):
    """
    Function to be iterated over a sparse user vectors Pandas DataFrame to produce naive Bayesian content-based predictions.
    Returns the equivalent pseudo-user's vector.
    """
    # Drop missing businesses to form labels y for training of the classifier.
    y = row.dropna()
    # Retrieve the corresponding business feature vectors.
    businessIndices = y.index
    X = businessVectors[businessIndices]

    # Fit the classifier to the user's content preferences.
    clf.fit(X, y)

    # Predict rating for all missing businesses. {
    businessIndicesSet = set(businessIndices)
    indicesToPredict = [i for i in range(totalBusinesses) if i not in businessIndicesSet]
    X = businessVectors[indicesToPredict]
    predictions = clf.predict(X)
    # }

    # Merge predictions into ratings vector to form the pseudo-user's vector.
    row.loc[indicesToPredict] = predictions
    return row

def kNNContentBasedPredict(row, businessSimilarities):
    """
    Function to be iterated over a sparse user vectors Pandas DataFrame to produce kNN content-based predictions.
    Returns the equivalent mean-adjusted pseudo-user's vector.
    """
    # Drop similarity entries for unrated businesses.
    businessSimilarities = businessSimilarities.where(row.notna()).dropna(axis=0)
    # Calculate mean-adjusted weighted sum of rated businesses to return a prediction for each business. {
    predictions = businessSimilarities.multiply(row.dropna(), axis=0)
    sumSimilarities = businessSimilarities.sum(axis=0).replace({0:1})
    predictions = predictions.sum(axis=0) / sumSimilarities
    # }
    # Merge predictions and ratings to form the pseudo-user's mean-adjusted vector.
    row = row.where(row.notna(), predictions)
    return row