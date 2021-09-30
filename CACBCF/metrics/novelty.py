import numpy as np

from CACBCF.model.collaborativeFiltering import predict

def testNovelty(df, businesses, businessesToIndex, hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, n=10, mode="hybrid"):
    """
    Function to calculate and return novelty.
    """
    if mode == "hybrid":
        novelty = pseudoUserVectors.apply(lambda row: calculateNovelty(row, df, businesses, hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, n=n, hybrid=True), axis=1)
    elif mode == "content":
        novelty = pseudoUserVectors.apply(lambda row: calculateNovelty(row, df, businesses, hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, n=n, contentOnly=True), axis=1)
    elif mode == "collaborative":
        novelty = pseudoUserVectors.apply(lambda row: calculateNovelty(row, df, businesses, hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, n=n, hybrid=False), axis=1)
    novelty = novelty.sum() / (len(novelty.index)*n)
    return novelty

def calculateNovelty(row, df, businesses, hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, n=10, hybrid=True, contentOnly=False):
    """
    Function to be iterated over a Pandas DataFrame with user_ids as its index to return sum of novelty per row (for the mean to then be calculated). 
    See paper Section II-E for details on novelty calculations.
    """
    userId = row.name
    userIndex = pseudoUserVectors.index.get_loc(userId)
    priorRatings = df[df.user_id == userId].business_id.unique()
    if not contentOnly:
        predictedRatings = predict(userIndex, hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, hybrid=hybrid)
    else:
        predictedRatings = row
    predictedRatings = predictedRatings[~businesses.iloc[predictedRatings.index].business_id.isin(priorRatings)]
    recommendations = predictedRatings.nlargest(n)
    novelty = np.sum(businesses.iloc[recommendations.index].popularityNegativeLog2)
    return novelty