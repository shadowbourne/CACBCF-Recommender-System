from CACBCF.UI.recommendUI import recommendUI
from CACBCF.train import train
from CACBCF.UI.submitReview import addReviewUI
def userOptionsUI(df, businesses, userId, userIndex, hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, businessesToIndex, state):
    """
    Main menu.
    """
    print("\n1) Submit a new review.")
    print("2) Retrieve recommendations.")
    print("3) Save and exit.")
    # print("4) Retrain recommender system. (WARNING: This may take some time.)")
    switchString = input("Please enter the appropriate number for the action you would like to perform:\n")
    failure = True
    while failure:
        failure = False
        switch = switchString.strip()
        if switch == "1":
            userPresent = True
            while userPresent:
                _, _, df = addReviewUI(userId, 0, df, businesses)
                print("Would you like to submit any more reviews?")
                print("1) Yes.")
                print("2) No.")
                switch2String = input("Please enter the appropriate number for the action you would like to perform:\n")
                failure = True
                while failure:
                    failure = False
                    switch2 = switch2String.strip()
                    if switch2 == "1":
                        userPresent = True
                        break
                    elif switch2 == "2":
                        userPresent = False
                        print("Retraining our model with your specific tastes in mind. This may take a while...")
                        hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, businessesToIndex, businesses = train(df, businesses, retrain=True, userId=userId, previousUserVectors=userVectors, previousPseudoUserVectors=pseudoUserVectors, previousHybridCorrelationWeights=hybridCorrelationWeights)   
                        print("Done!")
                        userIndex = userVectors.index.get_loc(userId)
                        break
                    else:
                        failure = True
                        switch2String = input("ERROR: \"{}\" is not a valid number, please try again:   ".format(switch2String))
        elif switch == "2":
            # Retrieve recommendations from the RS.
            df, businesses, hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, businessesToIndex = recommendUI(df, businesses, userId, userIndex, hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, businessesToIndex, state)
        elif switch == "3":
            print("Goodbye, we hope you had a lovely time, and hope to be of service to you again soon!")
            return False, df, businesses, userId, userIndex, hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, businessesToIndex

        # For admin use only. V
        # elif switch == "3":
        #     print("Retraining our model with your specific tastes in mind. This may take a while...")
        #     hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, businessesToIndex, businesses = train(df, businesses, retrain=True, userId=userId, previousUserVectors=userVectors, previousPseudoUserVectors=pseudoUserVectors, previousHybridCorrelationWeights=hybridCorrelationWeights, hybrid=True)    
        #     print("Done!")
        #     userIndex = userVectors.index.get_loc(userId)

        else:
            failure = True
            switchString = input("ERROR: \"{}\" is not a valid number, please try again:    ".format(switchString))
    failure = True
    while failure:
        failure = False
        print("\nIs there anything more we can help you with today?")
        print("1) Yes.")
        print("2) No.")
        answerString = input()
        answer = answerString.strip()
        # If the user is finished return False so that this function will not be re-entered.
        if answer == "1":
            print("\nHow can we be of further assistance {}?".format(userId))
            return True, df, businesses, userId, userIndex, hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, businessesToIndex
        elif answer == "2":
            return False, df, businesses, userId, userIndex, hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, businessesToIndex
        else:
            failure = True
            answerString = input("ERROR: \"{}\" is not a valid number, please try again:\n".format(answerString) )
   