# A Context Aware Restaurant Recommender System Using Content-Boosted Collaborative Filtering
Submitted as part of the degree of Msci Natural Sciences (3rd year) to the Board of Examiners in the Department of Computer Sciences, Durham University. 
This summative assignment was assessed and marked by the professor of the module in question:
## Grade: 1st - 92/100
## Paper Introduction:
This paper proposes a restaurant recommender system (RS) by
building on the works of various authors, with the aim of aiding
users in finding restaurants they will enjoy either in their local area
or further afield. 

Collaborative filtering (CF) (Herlocker et al. [1]),
content-based filtering (CB/CBF) and context aware (CA) (Zeng et
al. [2]) approaches are all utilised and combined in a hybrid scheme
(HS), similar to that introduced by Melville et al [3], which we name
CACBCF.

It will be apparent that RSs can become complex systems
made of many and differing components, as is well demonstrated in
the literature survey by Burke [4] which compares some 41 HSs.

Such systems result in a myriad of ethical issues, as surveyed and
investigated in the works of Milano et al. [5] and Germano et al. [6].
## Contents of repo:
* A Context Aware Restaurant Recommender System Using Content-Boosted Collaborative Filtering Paper - report.pdf.
* A 2 minute demo video of the Recommender System designed - demoVideo.mp4.
* Source code - CACBCF/, main.py & runMetrics.py.
* Pre-processed Yelp datasets for users and businesses.
## Instructions:
All scripts are designed to be run on Python 3.8.3 with the dependencies listed in requirements.txt, which can be installed via "pip install -r requirements.txt".
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
