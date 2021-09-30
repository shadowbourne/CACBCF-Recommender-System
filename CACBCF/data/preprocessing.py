import pandas as pd
import numpy as np
import json

def json_to_pandas(fileName, keys):
    """
    Function to take Yelp JSON files and return a Pandas DataFrame, constructed chunk-wise, with the corresponding keys.
    This done do as to handle massive inputs that cannot fit into RAM.
    """
    df = pd.DataFrame(columns=keys)
    miniDfArray = []
    print("Progress:")
    with open(fileName, encoding='UTF-8') as file:
        for i, line in enumerate(file):
            if i%1000000==0:
                print(i)
                if i%2500000==0 and i!=0:
                    miniDf = pd.DataFrame(miniDfArray)
                    df = df.append(miniDf)
                    miniDfArray = []
            contents = json.loads(line)
            contents = {k:contents[k] for k in keys}
            miniDfArray.append(contents)
    miniDf = pd.DataFrame(miniDfArray)
    df = df.append(miniDf)
    return df

def prep_data(fraction, state="QC", businessStarsThreshold=3., businessRatingThreshold=20, userRatingThreshold=5):
    """
    Function to prepare pre-processed dataset from Yelp JSON files.
    """
    df = json_to_pandas(fileName = "yelp_academic_dataset_review.json", keys = ["business_id", "user_id", "stars", "date"])

    businesses = json_to_pandas(fileName = "yelp_academic_dataset_business.json", keys = ["business_id", "review_count", "categories", "name", "address", "city", "state", "postal_code", "stars", "is_open", "hours", "latitude", "longitude"])

    businesses = businesses.dropna(axis=0, subset=["categories"])
    businesses = businesses[businesses.state == state]
    businesses = businesses[businesses.review_count >= businessRatingThreshold]
    businesses = businesses[businesses.stars >= businessStarsThreshold]
    businesses = businesses[businesses.categories.str.contains("Restaurant")]
    businesses = businesses.drop(["state"], axis=1)

    df = df.merge(businesses["business_id"], "inner", "business_id")

    users = json_to_pandas(fileName = "yelp_academic_dataset_user.json", keys = ["user_id","review_count"])

    users = users[users.review_count >= userRatingThreshold]
    users = users.drop("review_count", axis=1)

    df = df.merge(users, "inner", "user_id")

    # Only keep user's most recent rating per restaurant.
    df.date = pd.to_datetime(df.date)
    df = df.sort_values("date", ascending=False).drop_duplicates(["user_id", "business_id"],keep="first")
    df = df.drop(["date"], axis=1)
    
    users = users[users.user_id.isin(df.user_id.unique())]
    
    userMaskList = df.user_id.value_counts()
    userMaskList = userMaskList[userMaskList >= userRatingThreshold] 
    users = users[users.user_id.isin(userMaskList.index)]
    
    users = users.sample(frac=fraction)

    df = df.merge(users, "inner", "user_id")

    businessMaskList = df.business_id.value_counts()
    businessMaskList = businessMaskList[businessMaskList >= businessRatingThreshold] 
    businesses = businesses[businesses.business_id.isin(businessMaskList.index)]

    df = df.merge(businesses["business_id"], "inner", "business_id")

    businesses = businesses[businesses.business_id.isin(df.business_id.unique())]

    businesses.address = businesses.address + ", " + businesses.city + ", " + businesses.postal_code
    businesses = businesses.drop(["city", "postal_code"], axis=1)

    covid = json_to_pandas(fileName = "yelp_academic_dataset_covid_features.json", keys = ["business_id","delivery or takeout"])

    booleanDictionary = {'TRUE' : True, 'FALSE' : False}
    covid = covid.replace(booleanDictionary)
    covid = covid.rename({"delivery or takeout" : "covidTakeaway"}, axis=1)

    businesses = businesses.merge(covid, "inner", "business_id")

    businesses = businesses.reset_index()
    businesses = businesses.drop(["index"], axis=1)
    
    businessCodes, uniqueBusinesses = businesses.business_id.factorize()
    businessesToIndex = pd.Series(businessCodes, index=uniqueBusinesses)
    df = df.assign(business_idTempIntEncoding=businessesToIndex.loc[df.business_id].values)

    popularity = df.business_id.value_counts() / len(df.user_id.unique())
    popularity.index = businessesToIndex.loc[popularity.index].values
    businesses = businesses.assign(popularityNegativeLog2=-np.log2(popularity))
    
    businesses = businesses.assign(stars=df.groupby("business_id").mean().loc[businesses.business_id, "stars"].values)

    df.to_pickle("user.pkl")
    businesses.to_pickle("businesses.pkl")
    return df, businesses
