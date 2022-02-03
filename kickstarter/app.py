from flask import Flask, render_template
from .preprocess import test_func, wrangle
from .models import DB, Campaign
from .kickstarter import add_campaigns, sql_to_df
import pandas as pd
from sqlalchemy import select

def create_app():
    '''Function called in __init__.py to create our Flask App'''
    app = Flask(__name__)


    # Home route
    @app.route('/')
    def home_page():
        #test = test_func()
        #return f'{test}'
        return render_template('base.html', title='Kickstarter')
        #return render_template('index.html')
        #return 'This is the home page'

    # App config settings including the sqlite file name
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db_kickstarter.sqlite3'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # Connect our database to the app object
    DB.init_app(app)

    # Route to reset the database file from our processed dataframe
    @app.route('/reset')
    def reset():
        # Drop all rows from database and create new ones
        DB.drop_all()
        DB.create_all()
        # Run the add_campaigns function to grab all of the rows from our
        # dataframe and insert them into the database
        df = pd.read_csv(
            'https://raw.githubusercontent.com/FT-Kickstarter-05/Kickstarter/main/2018_ks_data.csv')
        add_campaigns(df)
        return 'Db file reset successful'

    @app.route('/query')
    def query():
        result = DB.session.query(Campaign).limit(10).all()
        df_test = sql_to_df(result)
        #test_o = result[0].eval('ID')
        #test_o = vars(result[1])
        # test_0 = pd.read_sql_query(result)
        # return f'{result}'
        return f'{df_test}'

    @app.route('/predict')
    def prediction():
        result = DB.session.query(Campaign).limit(10).all()
        df_test = sql_to_df(result)
        data = wrangle(df_test)
        
        predicted = test_func(data)
        # predicted = data.columns
        return f'Here are the first five predictions from our model that were\
                genearated from X_test: <br>{predicted}'

    return app
