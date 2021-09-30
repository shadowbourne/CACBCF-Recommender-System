import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def predict(userIndex, hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, hybrid=True):
    """
     This function acts as the collaborative filtering component and produces predictions accordingly.
     See paper Section III-B for full details.
    """
    if hybrid:
        # Compare active user a with all pseudo-users a^~ using adjusted cosine similarity.
        mask = ~userVectors.iloc[userIndex].isnull()
        similarities = cosine_similarity([userVectors.loc[:, mask].iloc[userIndex]], pseudoUserVectors.loc[:, mask])
        # similarities = cosine_similarity([pseudoUserVectors.iloc[userIndex]], pseudoUserVectors)

        # Calculate hybrid cosine correlation weights (see paper Section III-B).
        userWeights = (similarities * hybridCorrelationWeights[userIndex]).reshape(-1)
        # Calculate neighbourhood N.
        neighbourhood = np.argpartition(userWeights, -30)[-30:]
        # For self weighting. {
        neighbourhood = neighbourhood[neighbourhood != userIndex]
        neighbourhoodWeights = userWeights[neighbourhood]
        selfWeightingMax = max(neighbourhoodWeights)
        neighbourhood = np.append(neighbourhood, userIndex)
        neighbourhoodWeights = np.append(neighbourhoodWeights, selfWeightingMax * ratingCounts[userIndex] / 50 if ratingCounts[userIndex] < 50 else selfWeightingMax)
        # }
    else:
        # For collaborative-only baseline.
        similarities = cosine_similarity([pseudoUserVectors.iloc[userIndex]], pseudoUserVectors)
        useHybridCorrelationWeights = False
        if useHybridCorrelationWeights:
            userWeights = (similarities * hybridCorrelationWeights[userIndex]).reshape(-1)
        else:
            userWeights = similarities.reshape(-1)
        neighbourhood = np.argpartition(userWeights, -30)[-30:]
        neighbourhoodWeights = userWeights[neighbourhood]
        
    # Retrieve neighbourhood pseudo-user vectors.
    neighbourhoodVectors = pseudoUserVectors.iloc[neighbourhood]
    # Calculate mean-adjusted weighted sum. {
    weightedRatings = neighbourhoodVectors * neighbourhoodWeights.reshape(-1,1)
    predictions = weightedRatings.sum(axis=0) / neighbourhoodWeights.sum(axis=0) + meanUserRatings[userIndex]
    # }

    # If sum of weights = 0     as    all user submitted ratings = meanUserRatings[userIndex] (Extreme case handling)
    predictions = predictions.fillna(meanUserRatings[userIndex])

    return predictions