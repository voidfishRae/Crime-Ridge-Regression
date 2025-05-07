# Crime-Ridge-Regression
Analysis of the results of a ridge regression (via Scikit Learn) when applied to a dataset of community and crime statistics to predict likelihood of violent crime in different communities.

Dataset used:
http://archive.ics.uci.edu/ml/datasets/communities+and+crime

To run: ensure communities.txt and the main file are in the same folder, and run the python file. 
Output will show in the console window.

1. Refines the dataset - selects the relevant information from csv file and interprets it. This means cleaning up the data and setting the categories for the target and protected variables. All the data is numerical, so this step sorts the information into more general categories to allow meaningful comparison of results. 
2. Applies a ridge regression machine learning algorithm to the cleaned up data - from sciki-learn
 https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge.predict
3. Uses 10-fold cross-validation to evaluate the algorithm. 
4. Prints results and comparisons for different protected variable classes.
