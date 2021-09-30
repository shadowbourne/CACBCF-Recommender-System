import requests

def get_address(address):
    """
    Function to retrieve co-ordinates and a full address from MapQuest geocoding API by sending off a user-inputted address.
    """
    apiKey = "cjnRYfXH9Zv5vYT94e1hu5so2JhUAHT3"
    url = "https://www.mapquestapi.com/geocoding/v1/address?key={0}&location={1}&outFormat=json&maxResults=1&thumbMaps=false".format(apiKey, address)
    request = requests.get(url)
    if request.status_code != 200:
        return False, "GET request error code: " + request.status_code, None
    try:
        res = request.json()["results"]
        if len(res) == 0:
            return False, "No address returned!", None
        if len(res) != 1:
            print(str(len(res)) + " Adresses returned!")
        placeInfo = res[0]["locations"]
        if len(placeInfo) == 0:
            return False, "No address returned!", None
        if len(placeInfo) != 1:
            print(str(len(placeInfo)) + " Locations returned!")
        
    except:
        return False, "NOT JSON ADDRESS RESPONSE!", request.content, None
    try:
        addressString = ""
        keys = ["street", "adminArea6", "adminArea5", "adminArea4", "adminArea3", "adminArea1"]
        for key in keys:
            if placeInfo[0][key] != "":
                addressString += placeInfo[0][key] + ", "
        addressString = addressString[:-2]
        lat = placeInfo[0]["displayLatLng"]["lat"]
        lng = placeInfo[0]["displayLatLng"]["lng"]

        if (not lat and lat != 0) or (not lng and lng != 0):
            return False, "No address returned!", None
        else:
            return True, addressString, [lat, lng]
    except:
        return False, "No address returned!", None