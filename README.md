# Kickstarter Team Project

This app utilizes a dataset of 100,000 data points of kickstarter company details and implements a neural network model with a 91% accuracy score to predict the success status of a kickstarter company.

The team used advanced cleaning, preprocessing, feature engineering, encoding, splitting, and scaling methods to model and train a neural network project. This app uses early stopping to prevent some of the over-fitting problems associated with these kinds of models; the training stops if the validation loss is not improving for three epochs and returns to the best weights the model had. We used class weights to get better performance from the model, based on the skew of the imbalanced classes.

The trained model was saved and deployed on our custom flask app to generate and display predicted kickstart company success status to users.  To optimize processing speed, the initial dataset was converted to a mySQL database via MongoDB.  Our flask app is also deployed on Heroku for anyone to visit and view data visualizations and predictions for kickstarter companies.  

# Heroku APP

Deployed URL: https://kickstarter-05.herokuapp.com/

# Meet the Team

### Soloman A. DS-33 [Github](https://github.com/Solomansjib) [LinkedIn](https://www.linkedin.com/in/soloman-a/)

### John A. Baker Jr. DS-33 [Github](https://github.com/BakerJr1904) [LinkedIn](https://www.linkedin.com/in/john-a-baker-jr/)

### Bobby Wilt DS-34 [Github](https://github.com/BobbyWilt) [LinkedIn](https://www.linkedin.com/in/bobbywilt/)

The purpose of Build Week is to empower students to demonstrate mastery of your learning objectives. The Build Weeks experience helps prepare students for the job market.
