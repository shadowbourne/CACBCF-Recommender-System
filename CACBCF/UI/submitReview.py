def addReviewUI(userId, n_added, df, businesses):
    """
    Function to allow user to search for or select a business to submit a rating for.
    """
    badId = True
    while badId: 
        badId = False
        if n_added == 0:
            s = "a"
        else:
            s = "the next"
        print("Please submit the business id of {} restaurant you would like to review today!\n  Alternatively type SEARCH to search for the restaurant's corresponding business id or type EXIT if you are finished adding reviews (but remember the more the better!):".format(s))
        business_id = input()
        if "SEARCH" in business_id:
            noResults = True
            while noResults:
                noResults = False
                search = input("Please enter the name or part of the address of the restaurant you would like to search for, or EXIT if you would like to cancel your search:\n")
                if "EXIT" in search:
                    print(" ")
                    return True, n_added, df
                search = search.lower()
                # Search in name or address for a match.
                results = businesses[businesses.name.str.lower().str.contains(search) | businesses.address.str.lower().str.contains(search)][["name", "business_id", "address"]]
                i = 1
                # Print out results for user selection.
                for business in results.itertuples(index=False, name=None):
                    print("{}) {}, {}. (business_id: {})".format(i, business[0], business[2], business[1]))
                    i += 1
                if i == 1:
                    print("No search results found, please try again.\n")
                    noResults = True
                    continue
                badNumber = True
                while badNumber: 
                    badNumber = False
                    inputString = input("Please enter the number of the restaurant you would like to review, or EXIT if you would like to cancel your search:\n")
                    if "EXIT" in inputString:
                        print(" ")
                        return True, n_added, df
                    elif inputString.isdigit():
                        inputInt = int(inputString)
                        if inputInt < i and inputInt > 0:
                            business_id = results.iloc[inputInt - 1]["business_id"]
                            businessName = results.iloc[inputInt - 1]["name"]
                            _, n_added, df = submitReviewSubUI(business_id, businessName, userId, n_added, df)          
                            return True, n_added, df
                        else:
                            print("ERROR: {} is not an integer between 1 and {}.".format(inputString, i - 1))
                            badNumber = True
                    else:
                        print("ERROR: {} is not an integer between 1 and {}.".format(inputString, i - 1))
                        badNumber = True
            continue                    

        elif "EXIT" in business_id:
            print("Thank you for submitting your reviews!")
            return False, n_added, df

        elif business_id not in businesses.business_id.unique():
            print("ERROR: This restaurant does not exist. Please re-enter a valid business id. If unsure of the business id that correspsonds to your restaurant, please type SEARCH to search the Yelp database.\n") 
            badId = True
            continue
        # If business_id is valid by above if statement then call submitReviewSubUI.
        _, n_added, df = submitReviewSubUI(business_id, businesses[businesses.business_id == business_id].iloc[0]["name"], userId, n_added, df)          
        return True, n_added, df

def submitReviewSubUI(business_id, businessName, userId, n_added, df):
    """
    Function to allow user to submit a rating for a business.
    WARNING: business_id must be checked before being passed to this function.
    """
    badRating = True
    while badRating: 
        badRating = False
        stars = input("Please enter the number of stars you would like to give {}, or type EXIT to cancel: ( ? / 5.0 )\n(Warning: This data will be stored and reused (anonymously) for future predictions for yourself and others.)\n".format(businessName))
        if "EXIT" in stars:
            return False, n_added, df
        try:
            stars = float(stars)
        except ValueError:
            print("ERROR: {} is not a number between 0.0 and 5.0.".format(stars))
            badRating = True
            continue
        if stars <= 5 and stars >= 0:
            # If user has not rated this restaurant before append this review on to the end of the df.
            if business_id not in df[df.user_id == userId].business_id.unique():
                df.loc[df.index.max()+1] = [business_id, userId, stars, None]
            # Else, replace old review entry.
            else:
                df.loc[df[(df.user_id == userId) & (df.business_id == business_id)].index] = [business_id, userId, stars, None]
            print("Review submitted!\n")
            n_added += 1
            return True, n_added, df
        else:
            print("ERROR: {} is not a number between 0.0 and 5.0.".format(stars))
            badRating = True