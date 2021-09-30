import pandas as pd
from sklearn.metrics import confusion_matrix

from CACBCF.model.collaborativeFiltering import predict

def testUsage(trainDf, testDf, businesses, businessesToIndex, hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, n=10, mode="hybrid"):
    """
    Function to calculate and return F1-score, precision and recall.
    """
    # Calculate confusion matrices for each row.
    if mode == "hybrid":
        usage = pseudoUserVectors.apply(lambda row: calculateUsage(row, trainDf, testDf, businesses, hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, n=n, hybrid=True), axis=1)
    elif mode == "content":
        usage = pseudoUserVectors.apply(lambda row: calculateUsage(row, trainDf, testDf, businesses, hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, n=n, contentOnly=True), axis=1)
    elif mode == "collaborative":
        usage = pseudoUserVectors.apply(lambda row: calculateUsage(row, trainDf, testDf, businesses, hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, n=n, hybrid=False), axis=1)
    # Aggregate all confusion matrices into one matrix.
    confusionFrame = usage.sum()
    # Calculate precision, recall and F1-score.
    precision = confusionFrame["TP"] / (confusionFrame["TP"] + confusionFrame["FP"])
    recall = confusionFrame["TP"] / (confusionFrame["TP"] + confusionFrame["FN"])
    fScore = 2 * (precision * recall) / (precision + recall)
    return fScore, precision, recall

def calculateUsage(row, trainDf, testDf, businesses, hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, n=10, hybrid=True, contentOnly=False):
    """
    Function to be iterated over a Pandas DataFrame of pseudo-user vectors to calculate a confusion matrix per row
    with a true positive corresponding to an item appearing in top n recommendations and the item appearing in test set.
    """
    userId = row.name
    userIndex = pseudoUserVectors.index.get_loc(userId)
    priorRatings = trainDf[trainDf.user_id == userId].business_id.unique()
    if not contentOnly:
        predictedRatings = predict(userIndex, hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, hybrid=hybrid)
    else:
        predictedRatings = row
    predictedRatings = predictedRatings[~businesses.iloc[predictedRatings.index].business_id.isin(priorRatings)]
    recommendations = predictedRatings.nlargest(n)
    recommended = pd.Series(index=predictedRatings.index, data=predictedRatings.index.isin(recommendations.index))
    testPriorRatings = testDf[testDf.user_id == userId].business_id.unique()
    used = pd.Series(index=predictedRatings.index, data=businesses.iloc[predictedRatings.index].business_id.isin(testPriorRatings))
    confusionMatrix = confusion_matrix(used, recommended, labels=[True,False])
    row = pd.Series(index=["TP", "FP", "FN", "TN"], data=confusionMatrix.reshape(-1))
    return row