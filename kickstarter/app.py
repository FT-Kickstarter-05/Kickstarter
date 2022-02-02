from flask import Flask, render_template
from .preprocess import test_func
from .models import DB, Campaign
from .kickstarter import add_campaigns

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
    # @app.route('/reset')
    # def reset():
    #     # Drop all rows from database and create new ones
    #     DB.drop_all()
    #     DB.create_all()
    #     # Run the add_campaigns function to grab all of the rows from our
    #     # dataframe and insert them into the database
    #     add_campaigns()
    #     return 'Db file reset successful'

    @app.route('/predict')
    def prediction():
        predicted = test_func()
        return f'Here are the first five predictions from our model that were\
                genearated from X_test: <br>{predicted}'

    return app
