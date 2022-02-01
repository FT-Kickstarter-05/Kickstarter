from flask_sqlalchemy import SQLAlchemy


# Create a DB Object
DB = SQLAlchemy()


class Campaign(DB.Model):
    ''' This class will hold data for each row in the kickerstarter dataframe.
    The data structure is subject to change as run through preprocessing and
    feature engineering steps.'''
    tablename = 'campaign'
    ID = DB.Column(DB.BigInteger, primary_key=True, nullable=False)
    name = DB.Column(DB.String, nullable=True)  # has null values
    category = DB.Column(DB.String, nullable=False)
    main_category = DB.Column(DB.String, nullable=False)
    currency = DB.Column(DB.String, nullable=False)
    deadline = DB.Column(DB.String, nullable=False)
    goal = DB.Column(DB.Float, nullable=False)
    launched = DB.Column(DB.String, nullable=False)
    pledged = DB.Column(DB.Float, nullable=False)
    state = DB.Column(DB.String, nullable=False)
    backers = DB.Column(DB.Integer, nullable=False)
    country = DB.Column(DB.String, nullable=False)
    usd_pledged = DB.Column(DB.Float, nullable=True)  # has null values
    usd_pledged_real = DB.Column(DB.Float, nullable=False)
    usd_goal_real = DB.Column(DB.Float, nullable=False)
