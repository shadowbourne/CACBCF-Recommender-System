import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity

from CACBCF.model.contentBased import kNNContentBasedPredict, bayesianContentBasedPredict

def train(df, businesses, retrain=False, userId=None, previousUserVectors=None, previousPseudoUserVectors=None, previousHybridCorrelationWeights=None, contentMode="kNN", hybrid=True):
    """
    Prepare user profiles and create dense pseudo-user vectors matrix using either naive Bayesian or kNN Content based prediction.
    hybrid: dictates whether to use Content based prediction or not.
    """
    # Note: Future work, adapt this function to use the parameter retrain throughout,
    # and only update the contents of each return item for the user in question, as is done under the kNN section below.
    # This would drastically speedup retraining times after feedback supplied by user for larger datasets.

    # Create an integer encoder for business_ids to allow efficient indexing of all subsequent non Pandas arrays e.g.business feature vector similarities.
    # Gets recalculated each time, and/as elsewhere in code, Nones and NaNs fill part of this column (on purpose).
    businessCodes, uniqueBusinesses = businesses.business_id.factorize()
    totalBusinesses = len(uniqueBusinesses)
    businessesToIndex = pd.Series(businessCodes, index=uniqueBusinesses)
    df = df.assign(business_idTempIntEncoding=businessesToIndex.loc[df.business_id].values)

    # Calculate the popularity of each business and store negative log_2 of it for use in novelty calculations.
    popularity = df.business_id.value_counts() / len(df.user_id.unique())
    popularity.index = businessesToIndex.loc[popularity.index].values
    businesses = businesses.assign(popularityNegativeLog2=-np.log2(popularity))

    # Construct user vectors, mean user ratings and user rating counts.
    # If training from scratch, calculate all user-vectors.
    if retrain == False:
        userVectors = df.pivot_table(index="user_id", columns="business_idTempIntEncoding", values="stars")
    # If not, update only a singular user's vector: 
    else:
        newUserIdVector = df[df.user_id == userId].pivot_table(index="user_id", columns="business_idTempIntEncoding", values="stars")
        previousUserVectors.loc[userId] = newUserIdVector.loc[userId]
        previousUserVectors = previousUserVectors.sort_index()
        userVectors = previousUserVectors
    missingBusinesses = businessesToIndex[~businessesToIndex.isin(df.business_idTempIntEncoding.unique())].dropna().astype(int)
    meanUserRatings = userVectors.mean(axis=1)
    meanUserRatings = meanUserRatings.fillna(meanUserRatings.dropna().mean())
    userVectors = userVectors.reindex([*userVectors.columns.tolist() , *missingBusinesses.tolist()], axis=1) 
    ratingCounts = userVectors.count(axis=1)

    # Calculate hybrid correlation weights between all users (Melville et al. [3])
    M_u = np.where(ratingCounts < 100, ratingCounts/100, 1)
    u = len(M_u)
    if retrain == False:
        harmonicMeanWeights = np.array([[(M_u[i] * M_u[j]) / (M_u[i] + M_u[j]) for j in range(u)] for i in range(u)])
        hybridCorrelationWeights = userVectors.apply(lambda row: calculateHybridCorrelationWeights(row, userVectors, harmonicMeanWeights), axis=1)
        del M_u, harmonicMeanWeights
    else:
        harmonicMeanWeights = np.zeros((u,u))
        userIndex = userVectors.index.get_loc(userId)
        harmonicMeanWeights[userIndex] = [(M_u[userIndex] * M_u[j]) / (M_u[userIndex] + M_u[j]) for j in range(u)]
        # Work out new user's weights.
        newUserIdVector = userVectors.loc[userId:userId].apply(lambda row: calculateHybridCorrelationWeights(row, userVectors, harmonicMeanWeights), axis=1)

        # Update user's row.
        previousHybridCorrelationWeights.loc[userId] = newUserIdVector.loc[userId]
        previousHybridCorrelationWeights = previousHybridCorrelationWeights.sort_index()

        # Replace entries in all other rows (stupidly long way to do this :( ).
        previousHybridCorrelationWeights = (
                                        pd.DataFrame(previousHybridCorrelationWeights, index=previousHybridCorrelationWeights.index, columns=["weights"])
                                        .apply(lambda array: updateUserWeights(array, userIndex, newUserIdVector.loc[userId], userVectors), axis=1)
        )

        hybridCorrelationWeights = pd.Series(data=previousHybridCorrelationWeights.weights, index=previousHybridCorrelationWeights.index)


    # If content-based pseudo-vectors are desired.
    if hybrid:
        # Create Bag of Words business feature representations using TFIDF. {
        stop_words = ['bath', 'beauty', 'bed', 'bike','books', 'bookstores', 'bowls', 'estate' , 'education', 
                        'buildings', 'cards', 'caterers', 'centers', 'classes', 'clothing', 'consignment', 'convenience', 'cosmetics', 'couriers', 'court', 'de', 'event', 'gift', 'gifts', 'goods', 'government', 'groomers', 'hair', 'hobby', 'home', 'instruction','it',  'mags', 'maintenance', 'massage', 'mattresses', 'do',
                        'nail',  'pan', 'personal', 'planning',
                        'public',  'rentals', 'repair', 'resorts', 'restaurants', 'reunion', 'shared', 'shop', 'shops', 'sitting', 'spaces',  'spas', 'speakeasies', 'specialty', 'sporting', 'sports', 'sri', 'stands', 'stationery', 'stores',  'sum', 'supper', 'supplies', 'supply',  
                        'tours', 'travel', 'universities', 'used', 'vendors', 'venues', 'video', 'wear', 'yourself']
        # initialList = ['acai', 'active', 'afghan', 'african', 'airport', 'american', 'arabian', 'argentine', 'art', 'arts', 'asian', 'australian', 'austrian', 'bagels', 'bakeries', 'bangladeshi', 'bar', 'barbeque', 'bars', 'basque', 'bath', 'beauty', 'bed', 'beer', 'belgian', 'bike', 'bikes', 'biking', 'bistros', 'blues', 'books', 'bookstores', 'bowls', 
        #                 'brasseries', 'brazilian', 'breakfast', 'breweries', 'brewpubs', 'british', 'brunch', 'bubble', 'buffets', 'buildings', 'burgers', 'burmese', 'butcher', 'cabaret', 'cafes', 'cajun', 'cake', 'cakes', 'cambodian', 'canadian', 'candy', 'cantonese', 'cards', 'caribbean', 'caterers', 'centers', 'cheese', 'cheesesteaks', 'chefs', 'chicken', 'chinese', 'chips', 'chocolatiers', 'cinema', 'classes', 'clothing', 'clubs', 'cocktail', 'coffee', 'colleges', 'comedy', 'comfort', 'consignment', 'convenience', 'cooking', 'cosmetics', 'couriers', 'court', 'crafts', 'cream', 'creole', 'creperies', 'cuban', 'custom', 'czech', 'dance', 'day', 'decor', 'delicatessen', 'delis', 'de\'ethical', 'ethiopian', 'ethnic', 'european', 'event', 'falafel', 'farmers', 'fashion', 'fast', 'festivals', 'filipino', 'fish', 'fitness', 'flavor', 'flowers', 'fondue', 'food', 'free', 'french', 'frozen', 'fruits', 'furniture', 'fusion', 'galleries', 'game', 'garden', 'gastropubs', 'gay', 'german', 'gift', 'gifts', 'gluten', 'golf', 'goods', 'government', 'greek', 'grocery', 'groomers', 'hair', 'haitian', 'halal', 'hawaiian', 'health', 'himalayan', 'historical', 'hobby', 'home', 'hookah', 'hot', 'hotels', 'hungarian', 'iberian', 'ice', 'imported', 'indian', 'indonesian', 'instruction', 'international', 'internet', 'iranian', 'irish', 'it', 'italian', 'japanese', 'jazz', 'juice', 'karaoke', 'kebab', 'kitchen', 'korean', 'kosher', 'landmarks', 'lankan', 'laotian', 'latin', 'lebanese', 'life', 'live', 'local', 'lounges', 'macarons', 'mags', 'maintenance', 'malaysian', 'market', 'markets', 'massage', 'mattresses', 'mauritius', 'meat', 'mediterranean', 'mex', 'mexican', 'middle', 'modern', 
        #                 'mongolian', 'moroccan', 'mountain', 'music', 'musicians', 'nail', 'nepalese', 'new', 'nightlife', 'noodles', 'observatories', 'office', 'organic', 'pakistani', 'pan', 'party', 'patisserie', 'performing', 'persian', 'personal', 'peruvian', 'pet', 'pets', 'piano', 'pizza', 'planning', 'plates', 'poke', 'polish', 'portuguese', 'pot', 
        #                 'poutineries', 'pub', 'public', 'pubs', 'ramen', 'raw', 'real', 'rentals', 'repair', 'resorts', 'restaurants', 'reunion', 'roasteries', 'rooms', 'russian', 'salad', 'salons', 'salvadoran', 'sandwiches', 'scandinavian', 'schools', 'seafood', 'services', 'shacks', 'shared', 'shop', 'shopping', 'shops', 'singaporean', 'sitting', 'small', 'smokehouse', 'smoothies', 'social', 'soul', 'soup', 'southern', 'spaces', 'spanish', 'spas', 'speakeasies', 'specialty', 'spirits', 'sporting', 'sports', 'sri', 'stands', 'stationery', 'steakhouses', 'stores', 'street', 'sugar', 'sum', 'supper', 'supplies', 'supply', 'sushi', 'syrian', 'szechuan', 'tacos', 'taiwanese', 'tapas', 
        #                 'tea', 'tex', 'thai', 'theater', 'themed', 'tiki', 'tours', 'traditional', 'travel', 'trucks', 'turkish', 'universities', 'used', 'vegan', 'vegetarian', 'veggies', 'vendors', 'venezuelan', 'venues', 'video', 'vietnamese', 'vintage', 'waffles', 'wear', 'wedding', 'whiskey', 'wine', 'wineries', 'wings', 'women', 'yoga', 'yogurt', 'yourself']
        vectorizer = TfidfVectorizer(stop_words=stop_words)
        businessVectors = vectorizer.fit_transform(businesses.categories)
        # }

        if contentMode == "kNN": 
            # Calculate cosine similarity between business feature vectors.
            businessSimilarities = cosine_similarity(businessVectors, businessVectors)
            businessSimilarities = pd.DataFrame(data=businessSimilarities, index=range(totalBusinesses), columns=range(totalBusinesses))
            userVectors = userVectors.sub(meanUserRatings, axis=0)
            # If training from scratch, calculate all users pseudo-vectors.
            # (WARNING: depending on your dataset size this could quickly start to take days/weeks due to complexity. Hint, start small and increase dataset slowly, or use Bayes instead.)
            if retrain == False:
                pseudoUserVectors = userVectors.apply(lambda row: kNNContentBasedPredict(row,businessSimilarities), axis=1)
            # If not, and only updating a singular user's vector: 
            else:
                newUserIdVector = userVectors.loc[userId:userId].apply(lambda row: kNNContentBasedPredict(row,businessSimilarities), axis=1)
                previousPseudoUserVectors.loc[userId] = newUserIdVector.loc[userId]
                previousPseudoUserVectors = previousPseudoUserVectors.sort_index()
                pseudoUserVectors = previousPseudoUserVectors

        elif contentMode == "Bayes":
            # Create pseudo-user vectors using naive Bayesian classification.
            # This scales far better than kNN (e.g. 1 min vs 4 hours) and thus retraining was not implemented, especially as it is not used for our RS, only for baseline comparisons.
            # However, the code could be copied directly from above.
            clf = MultinomialNB()
            pseudoUserVectors = userVectors.round(0).apply(lambda row: bayesianContentBasedPredict(row,businessVectors,clf,totalBusinesses), axis=1)
            pseudoUserVectors = pseudoUserVectors.sub(meanUserRatings, axis=0)
            userVectors = userVectors.sub(meanUserRatings, axis=0)
        else:
            print("ERROR: contentMode \"{}\" not implemented, must be one of kNN or Bayes.".format(contentMode))
            exit(1)
        
    else:
        # If for collaborative-only baseline, impute data with user-means.
        userVectors = userVectors.sub(meanUserRatings, axis=0)
        pseudoUserVectors = userVectors.fillna(0)
    try:
        # Update mean business ratings.
        businesses = businesses.assign(stars=df.groupby("business_id").mean().loc[businesses.business_id, "stars"].values)
    except:
        # For when training for the train set for metric calculation, missing business_ids are no longer caught by the pipeline.
        pass
    return hybridCorrelationWeights, userVectors, pseudoUserVectors, meanUserRatings, ratingCounts, businessesToIndex, businesses

def calculateHybridCorrelationWeights(row, userVectors, harmonicMeanWeights): 
    """
    Function to be iterated over a Pandas DataFrame of user vectors 
    to build a significance weights matrix and add it to the harmonic mean weights
    to return a hybrid correlation weights matrix (Melville et al. [3]).
    """

    # Drop all items unrated by active user (this row) for all users. 
    mask = ~row.isnull()
    userVectors = userVectors.loc[:, mask]

    # Calculate the intersection (count) of rated items for each user with the active user.
    modR_anu = userVectors.count(axis=1) 

    # Apply significance weighting formula to build the matrix.
    significanceWeights = np.where(modR_anu < 50, modR_anu/50, 1)

    # Add significance weights and harmonic mean weights together to return hybrid correlation weights.
    userIndex = userVectors.index.get_loc(row.name)
    hybridCorrelationWeights = significanceWeights + harmonicMeanWeights[userIndex] 

    return hybridCorrelationWeights

def updateUserWeights(row, userIndex, newWeights, userVectors):
    """
    Function to update userIndex's weight entry in all other users' weight arrays by iterating over a Pandas DataFrame.
    """
    currentIndex = userVectors.index.get_loc(row.name)
    array = row.loc["weights"]
    array[userIndex] = newWeights[currentIndex]
    row.loc["temp"] = array
    return row