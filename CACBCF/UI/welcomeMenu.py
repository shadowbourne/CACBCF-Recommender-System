import os
import pickle
import pandas as pd
from CACBCF.data.preprocessing import prep_data
from CACBCF.train import train
from CACBCF.UI.mainMenu import userOptionsUI
from CACBCF.UI.submitReview import addReviewUI

def welcomeMenuUI():
    """
    Greeting function to load/prepare datasets and model and allow a user to login/register.
    Is the outermost function to start up the RS, and subsquently calls userOptionsUI.
    """
    state = "Quebec"
    # WARNING: if you change state code, for database to update you must delete user.pkl and (or) business.pkl and have the yelp dataset in our current directory. :)
    # The rest is handled for you (by prep_data below).
    stateCode = "QC"
    print("Hello!\nWelcome to the {} Restaurant Recommender System.".format(state))
    if os.path.exists("user.pkl") and os.path.exists("businesses.pkl"):
        print("Loading datasets...")
        df = pd.read_pickle("user.pkl")
        businesses = pd.read_pickle("businesses.pkl")
    else:
        print("Preparing datasets...")
        fraction = 1
        df, businesses = prep_data(fraction, state=stateCode)

    if os.path.exists("post-processed-user-data.pkl"):
        with open("post-processed-user-data.pkl", "rb") as f:
            hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, businessesToIndex = pickle.load(f)
    else:
        print("Training model...")
        hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, businessesToIndex, businesses = train(df, businesses, hybrid=True)    

    # To view some example user_ids to login with, comment in this line:
    # print(ratingCounts.nlargest(50))

    # Retrieve a user_id, login or registration.
    badId = True
    while badId:
        badId = False
        userId = input("If you are a returning customer, please enter your user id, or if not type REGISTER:\n")
        if "REGISTER" in userId:
            idTaken = True
            while idTaken:
                userId = input("Please enter the user id you would like to register an account with:\n")
                if userId in userVectors.index:
                    print("Sorry, this user id is already taken. :(\n")
                    continue
                idTaken = False
            print("\nWelcome user {} to the {} Restaurant Recommender System for the very first time! Please review a few of your favourite Yelp restaurants now so that we can custom tailor our model to you:".format(userId, state))
            userPresent = True
            n_added = 0
            while userPresent:
                userPresent, n_added, df = addReviewUI(userId, n_added, df, businesses)
                if n_added < 2:
                    print("\nPlease submit at least 2 reviews to get started! (Remember, the more you submit the better our recommendations will be.)\n")
                    userPresent = True
            print("Retraining our model with your specific tastes in mind. This may take a while...")
            hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, businessesToIndex, businesses = train(df, businesses, retrain=True, userId=userId, previousUserVectors=userVectors, previousPseudoUserVectors=pseudoUserVectors, previousHybridCorrelationWeights=hybridCorrelationWeights)    
            print("Done!\n")
            userIndex = userVectors.index.get_loc(userId)
            print("Welcome {}! How can we help you today?".format(userId))
            
        # Login section.
        elif userId in userVectors.index:
            userIndex = userVectors.index.get_loc(userId)
            print("\nWelcome back {}! How can we help you today?".format(userId))
        else:
            print("ERROR, user id not found, please try again.\n")
            badId = True
    
    # Go into main UI and remain until user selects the exit option.
    userPresent = True
    while userPresent:
        userPresent, df, businesses, userId, userIndex, hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, businessesToIndex = userOptionsUI(df, businesses, userId, userIndex, hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, businessesToIndex, state)

    # Once user has exited save all data to disk.
    with open("post-processed-user-data.pkl", "wb") as f:
        pickle.dump([hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, businessesToIndex], f)       
    df.to_pickle("user.pkl")
    businesses.to_pickle("businesses.pkl")
