# CACBCF-RS
All scripts are designed to be run on Python 3.8.3 with the dependencies listed in requirements.txt, which can be installed via "pip install -r requirements.txt".
## Instructions:
- To start the recommender system (CACBCF), first make sure either pre-processed pickled user and business datasets (user.pkl & businesses.pkl),
or the entire Yelp Open Dataset in json form, are inside the same file directory as main.py.

- Then run main.py (python main.py) to enter the main user interface. 
  * If either user.pkl or businesses.pkl is not present then they will be prepared from scratch from the Yelp Dataset (~10 mins) (which you must download from https://www.yelp.com/dataset). 
  * Then, if post-processed-user-data.pkl is not present (which can be located within and extracted from post-processed-user-data.rar), the user profiles and model will be trained from the above (~15 mins).
  * From this point onwards, any retraining (e.g. user feedback) will be take a second or 2. 
    
  * For a list of sample users to login as, either inspect users.pkl contents or use one of the following supplied examples, or instead register yourself as a new user!
    Sample user_ids:
    
    - 3aYeG-x5A44GIgmBHrwyAA (227 reviews)
    - U1vl4SQzO3wTAWlYVnSjnw (163 reviews)
    - ZD76B53WiEdv3g2lNgTbNg (112 reviews)
    - sKVpHfhkG_Nvgf_Vfb91Cg (88 reviews)
    - CebjpVd3PsofCgotWp60pg (74 reviews)
    - 5ca2MkCJFAMafDrRxLMlXQ (44 reviews)
    - ygjIo5gLQ8wmsOcTDiHG2Q (29 reviews)
    - 1uUwbiQfayJiGhqx3CjjIg (11 reviews)
    - C5faJHojrEUqX5VMFrsz3Q (5 reviews)

  * For sample locations, either pick anywhere in Quebec or choose from any of the following examples:

     - Boulevard Saint-Laurent, Montreal
     - H9B 1P7
     - Brossard
    
- Alternatively, run runMetrics.py (python runMetrics.py) to run metrics (WARNING: on the current dataset this may take a few hours).
