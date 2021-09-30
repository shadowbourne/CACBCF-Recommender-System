from CACBCF.model.collaborativeFiltering import predict

def testRMSE(testDf, businessesToIndex, hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, mode="hybrid"):
    """
    Function to calculate and return RMSE.
    """
     # Construct a user vectors matrix of test ratings to be iterated over by calculateSE to return sum of squared errors (SE).
    testDf = testDf.assign(business_idTempIntEncoding=businessesToIndex.loc[testDf.business_id].values)
    testUserVectors = testDf.pivot_table(index="user_id", columns="business_idTempIntEncoding", values="stars")
    if mode == "hybrid":
        # Hybrid scheme based predictions.
        summedSquaredError = testUserVectors.apply(lambda row: calculateSE(row, hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts), axis=1)
    elif mode == "content":
        # Content based predictions only.
        pseudoUserVectors = pseudoUserVectors.add(meanUserRatings, axis=0)
        summedSquaredError = ( (testUserVectors - pseudoUserVectors) ** 2 ).sum(axis=1)
    elif mode == "collaborative":
        # Collaborative based predictions only.
        summedSquaredError = testUserVectors.apply(lambda row: calculateSE(row, hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, hybrid=False), axis=1)
    # Sum the sums of RMSE per row, divide by total number of ratings to calculate mean, then square root.
    RMSE = ( summedSquaredError.sum() / len(testDf.index) ) ** 0.5
    return RMSE

def calculateSE(row, hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, hybrid=True):
    """
    Function to be iterated over a Pandas DataFrame of user vectors filled with test ratings only.
    """
    actualRatings = row.dropna()
    userId = actualRatings.name
    userIndex = pseudoUserVectors.index.get_loc(userId)
    predictedRatings = predict(userIndex, hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, hybrid=hybrid)
    # Calculate and return the sum of the squared errors between test ratings and predictions.
    summedSquaredError = ((actualRatings - predictedRatings) ** 2).dropna().sum()
    return summedSquaredError