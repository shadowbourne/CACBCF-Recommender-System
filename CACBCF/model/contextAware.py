import pandas as pd
from haversine import haversine_vector

from CACBCF.model.collaborativeFiltering import predict
from CACBCF.UI.submitReview import submitReviewSubUI
from CACBCF.train import train

def recommend(noRecommendations, covidTakeawayOnly, noveltyMode, geographicalContextAware, coords, D, businesses, userIndex, userId, hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, businessesToIndex, df, hybrid=True):
    """
     This function acts as an interactive context aware recommender taking all user preferences into account.
    """

    # Counter to be used to show what recommendation to start at.
    i = 1
    printNext = True
    repredict = True
    retrain = False
    # Store business_ids that user has previously rated, for use in weighting/excluding 
    startingRatings = df[df.user_id == userId].business_id.unique()
    # Loop to print the top [i, i + noRecommendations) recommendations 
    while printNext:
        printNext = False
        # If predictions must be remade e.g. radius R has been expanded or feedback has been provided.
        if repredict:
            # If user has submitted a new rating (feedback) since last iteration of the loop then retrain model.
            if retrain:
                hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, businessesToIndex, businesses = train(df, businesses, retrain=True, userId=userId, previousUserVectors=userVectors, previousPseudoUserVectors=pseudoUserVectors, previousHybridCorrelationWeights=hybridCorrelationWeights)
            predictions = predict(userIndex, hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, hybrid=hybrid)
            priorRatings = df[df.user_id == userId]
            s = pd.Series(index=businessesToIndex.loc[priorRatings.business_id].values, data=priorRatings.stars.values)
            # NOTE: weightedPredictions are used to sort predictions to form the top noRecommendations recommendations, while predictions are the predicted ratings shown to the user for these recommendations.
            if noveltyMode == "weighted":
                # Discount restaurants rated before start of session by 0.8.
                weightedPredictions = predictions.append(s[businesses.iloc[s.index].business_id.isin(startingRatings)] * 0.8)
                # Newly rated restaurants this session provided in the form of feedback remain untouched, however feedback on old restaurants is still discounted above.
                weightedPredictions = weightedPredictions.append(s[~businesses.iloc[s.index].business_id.isin(startingRatings)])
                # Replace predictions with prior ratings and feedback.
                weightedPredictions = weightedPredictions[~weightedPredictions.index.duplicated(keep='last')]
            else:
                # Replace predictions with prior ratings and feedback.
                weightedPredictions = predictions.append(s)
                weightedPredictions = weightedPredictions[~weightedPredictions.index.duplicated(keep='last')]
            predictions = predictions.append(s)
            predictions = predictions[~predictions.index.duplicated(keep='last')]
            if geographicalContextAware:
                businessCoords = businesses.iloc[predictions.index][["latitude","longitude"]].values.tolist()
                distances = pd.Series(index=predictions.index, data=haversine_vector([coords]*len(businessCoords), businessCoords, unit="km"))
                # Filter out restaurants further than D km away.
                mask = distances <= D
                distances = distances[mask]
                predictions = predictions[mask]
                weightedPredictions = weightedPredictions[mask]
                # Add "vicinity weighting" for sorting of predictions (see paper Section III-B).
                weightedPredictions = weightedPredictions + 1 * (1 - distances / D)
            repredict = False
       

        # Remove previously rated restaurants (before this session) if the user has indicated they would like to see only new restaurants.
        if noveltyMode == "novel-only":
            predictions = predictions[~businesses.iloc[predictions.index].business_id.isin(startingRatings)]
            weightedPredictions = weightedPredictions.loc[predictions.index]
        # Filter restaurants to only those delivering during the COVID-19 pandemic.
        if covidTakeawayOnly:
            predictions = predictions[businesses.iloc[predictions.index].covidTakeaway]
            weightedPredictions = weightedPredictions.loc[predictions.index]

        # If no filtered results remain.
        if len(predictions.index) == 0:
            print("Sorry, we have found no restaurants to recommend within the radius you selected, would you like it to be expanded to {}?".format(D * 2))
            print("If so, type 1, or type 0 to return to the main menu:")
            failure = True
            switch = input()
            while failure:
                failure = False
                if switch.strip() == "0":
                    print("")
                    break   
                elif switch.strip() == "1":
                    printNext = True
                    repredict = True
                    D *= 2
                    print("")
                    break   
                else:
                    failure = True
                    switch = input("ERROR: \"{}\" is not a valid number, please try again:    ".format(switch) )
        # If some filtered results remain.
        else:
            # Retrieve top [i, i + noRecommendations) predictions based on weightedPredictions.
            recommendations = predictions.loc[weightedPredictions.nlargest(noRecommendations + i - 1).iloc[(i - 1):].index]

            # To be used in inner loop to move recommendation list back 1 iteration.
            if len(recommendations.index) < noRecommendations:
                tempNoRecommendations = len(recommendations.index)
            else:
                tempNoRecommendations = noRecommendations
            if tempNoRecommendations == 0:
                print("That's all the recommendations we have for now!\n")
                break

            if geographicalContextAware:
                currentDistances = distances.loc[recommendations.index]
            results = businesses.iloc[recommendations.index][["name", "address", "stars"]]
            # Present recommendations to the user.
            for business, rating in zip(results.itertuples(index=False, name=None), recommendations.iteritems()):
                if geographicalContextAware:
                    distanceString = ", {:.2f}km away".format(currentDistances.iloc[i%noRecommendations - 1])
                else:
                    distanceString = ""
                print("{0}) {1}, {2}{5}. (What we think you will think: {4:.2f}, what others think: {3:.2f})".format(i, business[0], business[1], business[2], rating[1], distanceString))
                i += 1
            print("\nWould you like to see any more details of the recommended restaurants / submit any alterations to our predicted ratings to improve our understanding of your personal tastes?")
            print("If so, type the appropriate number, otherwise type -1 to see our next {} recommendations or, if you're happy and don't want to see anything else, type 0:".format(noRecommendations))
            failure = True
            switch = input()
            while failure:
                failure = False
                # Print next noRecommendations recommendations.
                if switch.strip() == "-1":
                    printNext = True
                    print("")
                    break
                elif switch.strip().isdigit():
                    num = int(switch.strip())
                    # Exit.
                    if num == 0:
                        break
                    # Show "More Details" for restaurant num.
                    elif num >= i - tempNoRecommendations and num < i:
                        if geographicalContextAware:
                            distanceString = "    Distance: {:.2f}km.".format(currentDistances.iloc[num%noRecommendations - 1])
                        else:
                            distanceString = ""
                        business =  businesses.iloc[recommendations.index[num%noRecommendations - 1]]
                        print("\nName: {}.    Address: {}.{}".format(business["name"], business.address, distanceString))
                        print("\nCategories and services (These are the tags we use to help extrapolate your content preferences):\n{}.".format(business.categories))
                        print("\nOpening hours: {}.".format(business.hours))
                        print("\nAverage rating: {:.2f}.   ".format(business.stars), end="")
                        print("Review count: {}.".format(business.review_count))
                        print("Our personal predicted rating for you: {:.2f}.".format(recommendations.iloc[num%noRecommendations - 1]))
                        print("\nIf you would like to submit feedback on this recommendation by submitting a rating, press 1, or to return to the recommendations list press 0:")
                        switchString = input()
                        failure = True
                        while failure:
                            failure = False
                            switch = switchString.strip()
                            # If user wants to submit feedback.
                            if switch == "1":
                                changed_, _, df = submitReviewSubUI(business.business_id, business["name"], userId, 0, df)
                                printNext = True
                                # If user submitted review then retrain model and show new recommendations.
                                if changed_ and not repredict:
                                    repredict = True
                                    retrain = True
                                    i = 1
                                    print("Taking into account your feedback...")
                                # Otherwise return to same place in predictions list.
                                else: 
                                    i -= tempNoRecommendations
                            # Return to same place in predictions list.
                            elif switch == "0":
                                printNext = True
                                i -= tempNoRecommendations
                                break
                            else:
                                failure = True
                                switchString = input("ERROR: \"{}\" is not a valid number, please try again:    ".format(switchString) )
                    else:
                        failure = True
                        switch = input("ERROR: \"{}\" is not a valid integer between {} and {}, please try again:    ".format(switch, i - tempNoRecommendations, i - 1))
                else:
                    failure = True
                    switch = input("ERROR: \"{}\" is not a valid integer between {} and {}, please try again:    ".format(switch, i - tempNoRecommendations, i - 1))
    return df, businesses, hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, businessesToIndex
