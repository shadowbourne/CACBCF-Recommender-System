from sklearn.utils import shuffle

from CACBCF.model.contextAware import recommend
from CACBCF.helpers import get_address

def recommendUI(df, businesses, userId, userIndex, hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, businessesToIndex, state):
    """
    Function to gather user's preferences for the context aware recommender to filter CF predictions and show recommendations to the user.
    """
    noRecommendationsString = input("\nHow many restaurants are you interested in being shown today?\n")
    noRecommendations = -1
    failure = True
    while failure:
        failure = False
        if noRecommendationsString.isdigit():
            noRecommendations = int(noRecommendationsString)
        else:
            failure = True
            noRecommendationsString = input("ERROR: \"{}\" is not a valid integer, please try again:    ".format(noRecommendationsString))

    print("\nDue to the current COVID sitation in your area, it may not be possible to visit restaurants in person.\nWould you therefore like to limit your search to only those offering a takeaway or delivery service during the pandemic?")
    print("1) Yes.")
    print("2) No, I am able to eat out.")
    switch2String = input("\nPlease enter the appropriate number for your choice:\n")
    covidTakeawayOnly = False
    failure = True
    while failure:
        failure = False
        switch2 = switch2String.strip()
        if switch2 == "1":
            covidTakeawayOnly = True
        elif switch2 == "2":
            covidTakeawayOnly = False
        else:
            failure = True
            switch2String = input("ERROR: \"{}\" is not a valid number, please try again:   ".format(switch2String))

    print("\nWould you like to include restaurants you have visited before as possible recommendations?")
    print("1) Yes.")
    print("2) Yes, but I would prefer to see new restaurants.")
    print("3) No, I would only like to see new restaurants.")
    switch3String = input("\nPlease enter the appropriate number for your choice:\n")
    noveltyMode = ""
    failure = True
    while failure:
        failure = False
        switch3 = switch3String.strip()
        if switch3 == "1":
            noveltyMode = "mixed"
        elif switch3 == "2":
            noveltyMode = "weighted"
        elif switch3 == "3":
            noveltyMode = "novel-only"
        else:
            failure = True
            switch3String = input("ERROR: \"{}\" is not a valid number, please try again:   ".format(switch3String))

    print("\nFinally, are you looking for local recommendations, or are you interested in state wide restaurants?")
    print("1) Yes, I would prefer local restaurants.")
    print("2) No, I do not mind where they are located within {}.".format(state))
    switch4String = input("\nPlease enter the appropriate number for your choice:\n")
    geographicalContextAware = False
    failure = True
    while failure:
        failure = False
        switch4 = switch4String.strip()
        if switch4 == "1":
            geographicalContextAware = True
        elif switch4 == "2":
            geographicalContextAware = False
        else:
            failure = True
            switch4String = input("ERROR: \"{}\" is not a valid number, please try again:   ".format(switch4String))

    address, coords = "", []
    if geographicalContextAware:
        success = False
        while not success:
            addressInput = input("\nPlease enter your address:\n")
            # Send address off to external geocoding services (MapQuest).
            success, address, coords = get_address(addressInput)
            if not success:
                print("ERROR: {} Please try again, perhaps with a more specific address?\n".format(address))
                continue
            print("Address found! Is this the correct address?")
            print(address)
            print("1) Yes.")
            print("2) No.")
            print("3) I would like to cancel and proceed with a state wide search instead.")
            switch5String = input("\nPlease enter the appropriate number for your choice:\n")
            failure = True
            while failure:
                failure = False
                switch5 = switch5String.strip()
                if switch5 == "1":
                    success = True
                elif switch5 == "2":
                    success = False
                elif switch5 == "3":
                    success = True
                    geographicalContextAware = False
                else:
                    failure = True
                    switch5String = input("ERROR: \"{}\" is not a valid number, please try again:   ".format(switch5String))         

    distance = -1
    if geographicalContextAware:
        print("\nIdeally, what is the maximum distance you would travel to eat? (km)")
        badNumber = True
        while badNumber: 
            badNumber = False
            distance = input()
            try:
                distance = float(distance)
                if distance > 0:
                    badNumber = False
                    D = distance
                else:
                    print("ERROR: {} is not a distance greater than 0.0 (in kilometres).".format(distance))
                    badNumber = True
            except ValueError:
                print("ERROR: {} is not a distance greater than 0.0 (in kilometres).".format(distance))
                badNumber = True
    else:
        D = 0

    print("\nThank you for inputting all of your preferences!")
    print("Please wait while we tailor our model to your tastes...\n")
    # Sample some of the user's favourite restaurants.
    if len(df[df.user_id == userId].index) < 40: 
        topNIds = df[df.user_id == userId].nlargest(4, columns="stars").business_id
        try:
            top2Ids = topNIds.sample(2)
        # When a user has less than 2 ratings.
        except ValueError:
            top2Ids = topNIds
    else: 
        topNIds = df[df.user_id == userId].nlargest(len(df[df.user_id == userId].index)//10, columns="stars").business_id
        top2Ids = topNIds.sample(2)
    top2 = businesses.name[businesses.business_id.isin(top2Ids)].tolist()
    try:
        print("We do this by first analysing your favourite restaurants, such as {} and {}.".format(top2[0], top2[1]))
    except:
        try:
            print("We do this by first analysing your favourite restaurants, such as {}.".format(top2[0]))
        except:
            print("We do this by first analysing your favourite restaurants.")
    

    # Randomly sample from top tags with respect to occurrence ratios.
    tags = businesses.categories[businesses.business_id.isin(topNIds)].tolist()
    tags = [t for tag in tags for t in tag.split(", ") if t not in ["Restaurants", "Food", "(New)", "Stores", "Flowers & Gifts"]]  
    tags = shuffle(tags)
    seen = set()
    duplicateTagsDropped = [x for x in tags if not (x in seen or seen.add(x))]
    topTags = duplicateTagsDropped[:3]
    
    print("We then try and find similar dining experiences, using restaurant tags from Yelp and comparing them to all the restaurants you have liked before, to predict how these restaurants may mirror your tastes.")
    try:
        print("For example, we notice that a lot of your favourite restaurants contain tags like {}, {} and {}.".format(topTags[0], topTags[1], topTags[2]))
    except: 
        try:
            print("For example, we notice that a lot of your favourite restaurants contain tags like {} and {}.".format(topTags[0], topTags[1]))
        except:
            try:
                print("For example, we notice that a lot of your favourite restaurants contain tags like {}.".format(topTags[0]))
            except:
                pass

    print("This is the content stage in our content boosted collaborative recommender system.")
    print("Finally, using collaborative filtering, we compare your extrapolated tastes with those of others, and use a weighted combination of the ratings of the most similar users to predict which restaurants you may enjoy.")

    print("\nDone! Here are our recommendations for you:")
    df, businesses, hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, businessesToIndex = recommend(noRecommendations, covidTakeawayOnly, noveltyMode, geographicalContextAware, coords, D, businesses, userIndex, userId, hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, businessesToIndex, df)
    return df, businesses, hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, businessesToIndex
   