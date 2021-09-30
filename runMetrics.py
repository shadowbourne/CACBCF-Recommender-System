from CACBCF.data.preprocessing import prep_data
import pickle
import pandas as pd

from CACBCF.train import train
from CACBCF.metrics.novelty import testNovelty
from CACBCF.metrics.accuracy import testRMSE
from CACBCF.metrics.usage import testUsage

def runMetrics(df, businesses):
    """
    Function to run and print out all metrics.
    """
    print("Running Novelty Metrics...\n")
    prep_data(1)
    hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, businessesToIndex, businesses = train(df, businesses, hybrid=False)
    collaborativeBaselineNovelty = testNovelty(df, businesses, businessesToIndex, hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, n=10, mode="collaborative")

    hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, businessesToIndex, businesses = train(df, businesses, hybrid=True)

    # Uncomment to save the trained model (as the above is the same as usual data prep).
    with open("post-processed-user-data.pkl", "wb") as f:
        pickle.dump([hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, businessesToIndex], f) 

    novelty_NBA = testNovelty(df, businesses, businessesToIndex, hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, n=10, mode="hybrid")
    NBAContentBaselineNovelty = testNovelty(df, businesses, businessesToIndex, hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, n=10, mode="content")

    hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, businessesToIndex, businesses = train(df, businesses, hybrid=True, contentMode="Bayes")
    novelty_Bayesian = testNovelty(df, businesses, businessesToIndex, hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, n=10, mode="hybrid")
    BayesianContentBaselineNovelty = testNovelty(df, businesses, businessesToIndex, hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, n=10, mode="content")

    print("Hybrid recommender Novelty Score (Bayesian CBF): {:.3}\nHybrid recommender Novelty Score (Neighbourhood-based CBF): {:.3}\nBaseline Novelty Score (Bayesian Content only): {:.3}\nBaseline Novelty Score (Neighbourhood-based Content only): {:.3}\nBaseline Novelty Score (Collaborative only): {:.3}".format(novelty_Bayesian, novelty_NBA, BayesianContentBaselineNovelty, NBAContentBaselineNovelty, collaborativeBaselineNovelty))

    print("\nRunning Accuracy and Usage Metrics...\n")

    # Split dataset into test and train by user (20:80 roughly).
    testDfIndices = df.groupby('user_id').apply(lambda s: s.sample(len(s.index)//5, random_state=642)).index.get_level_values(1)
    trainDfIndices = df.index[~df.index.isin(testDfIndices)]
    testDf = df.loc[testDfIndices]
    trainDf = df.loc[trainDfIndices]

    # OR split dataset into test and train (20:80).
    # trainDf, testDf = train_test_split(df, test_size=0.2, random_state=642)

    hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, businessesToIndex, businesses = train(trainDf, businesses, hybrid=False)
    collaborativeBaselineRMSE = testRMSE(testDf, businessesToIndex, hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, mode="collaborative")
    collaborativeBaselineF1, collaborativeBaselinePrecision, collaborativeBaselineRecall = testUsage(trainDf, testDf, businesses, businessesToIndex, hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, n=10, mode="collaborative")

    hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, businessesToIndex, businesses = train(trainDf, businesses)       
    RMSE_NBA = testRMSE(testDf, businessesToIndex, hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, mode="hybrid")
    F1_NBA, precision_NBA, recall_NBA = testUsage(trainDf, testDf, businesses, businessesToIndex, hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, n=10, mode="hybrid")
    NBAContentBaselineRMSE = testRMSE(testDf, businessesToIndex, hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, mode="content")
    NBAContentBaselineF1, NBAContentBaselinePrecision, NBAContentBaselineRecall = testUsage(trainDf, testDf, businesses, businessesToIndex, hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, n=10, mode="content")

    hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, businessesToIndex, businesses = train(trainDf, businesses, contentMode="Bayes")
    RMSE_Bayesian = testRMSE(testDf, businessesToIndex, hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, mode="hybrid")
    F1_Bayesian, precision_Bayesian, recall_Bayesian = testUsage(trainDf, testDf, businesses, businessesToIndex, hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, n=10, mode="hybrid")
    BayesianContentBaselineRMSE = testRMSE(testDf, businessesToIndex, hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, mode="content")
    BayesianContentBaselineF1, BayesianContentBaselinePrecision, BayesianContentBaselineRecall = testUsage(trainDf, testDf, businesses, businessesToIndex, hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, n=10, mode="content")

    print("Hybrid recommender RMSE (Bayesian CBF): {:.3}\nHybrid recommender RMSE (Neighbourhood-based CBF): {:.3}\nBaseline RMSE (Bayesian Content only): {:.3}\nBaseline RMSE (Neighbourhood-based Content only): {:.3}\nBaseline RMSE (Collaborative only): {:.3}\n".format(RMSE_Bayesian, RMSE_NBA, BayesianContentBaselineRMSE, NBAContentBaselineRMSE, collaborativeBaselineRMSE))
    print("Hybrid recommender F-1 Score (Bayesian CBF): {:.3}\nHybrid recommender F-1 Score (Neighbourhood-based CBF): {:.3}\nBaseline F-1 Score (Bayesian Content only): {:.3}\nBaseline F-1 Score (Neighbourhood-based Content only): {:.3}\nBaseline F-1 Score (Collaborative only): {:.3}\n".format(F1_Bayesian, F1_NBA, BayesianContentBaselineF1, NBAContentBaselineF1, collaborativeBaselineF1))

    # Save metric results.
    with open("results.pkl", "wb") as f:
        pickle.dump([novelty_Bayesian, novelty_NBA, BayesianContentBaselineNovelty, NBAContentBaselineNovelty, collaborativeBaselineNovelty, RMSE_Bayesian, RMSE_NBA, BayesianContentBaselineRMSE, NBAContentBaselineRMSE, collaborativeBaselineRMSE, F1_Bayesian, F1_NBA, BayesianContentBaselineF1, NBAContentBaselineF1, collaborativeBaselineF1], f) 

if __name__ == "__main__":
    df = pd.read_pickle("user.pkl")
    businesses = pd.read_pickle("businesses.pkl")
    runMetrics(df, businesses)