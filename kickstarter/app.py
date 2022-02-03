from flask import Flask, render_template
from .preprocess import test_func, wrangle
from .models import DB, Campaign
from .kickstarter import sql_to_df
# from .kickstarter import add_campaigns
import json


def create_app():
    '''Function called in __init__.py to create our Flask App'''
    app = Flask(__name__)

    # App config settings including the sqlite file name
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db_kickstarter.sqlite3'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # Connect our database to the app object
    DB.init_app(app)

    # Home route
    @app.route('/')
    def home_page():
        return render_template('base.html', title='Kickstarter')

    # Route to reset the database file from our processed dataframe.
    # Commenting this route out since it will crash heroku and our database
    # doesn't need to be updated
    # @app.route('/reset')
    # def reset():
    #     # Drop all rows from database and create new ones
    #     DB.drop_all()
    #     DB.create_all()
    #     # Run the add_campaigns function to grab all of the rows from our
    #     # dataframe and insert them into the database
    #     df = pd.read_csv(
    #         'https://raw.githubusercontent.com/FT-Kickstarter-05/Kickstarter/main/2018_ks_data.csv')
    #     add_campaigns(df)
    #     return 'Db file reset successful'

    @app.route('/plots')
    def query():
        return render_template('plots.html', title='plots')

    @app.route('/predict')
    def prediction():
        # Query first 10 rows of dataframe
        result = DB.session.query(Campaign).limit(10).all()
        # Convert SQL objects to a dataframe
        df_test = sql_to_df(result)
        # Reformat queried dataframe to be compatible with model
        data = wrangle(df_test)
        # Split data into X and y and run the model predictions
        predicted = test_func(data)
        # Reformat ouput to json
        predicted_json = json.dumps(predicted)
        return predicted_json

    return app
