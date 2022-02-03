from .models import DB, Campaign
import pandas as pd


def add_campaigns(df):
    '''This function attempt to add in each row of the kickstarter dataframe
    into a Campaign class.  If no errors are found, then it will add each
    Campaign class to a database file and commit the changes.'''

    try:
        # Convert dataframe to dictionary
        df_dict = df.to_dict('records')

        # Run a bulk insert into the database file db_kickstarter.sqlite3.
        # This will use the schema in the Campaign class within models.py
        DB.engine.execute(Campaign.__table__.insert(), df_dict)

    except Exception as error:
        print(f'Error resetting database {df}: {error}')
        raise error

def sql_to_df(obj_list):
    '''Takes in a list of SQL class objects and converts them to a dataframe'''
    cols = ['ID', 'name', 'category', 'main_category', 'currency', 'deadline',
       'goal', 'launched', 'pledged', 'state', 'backers', 'country',
       'usd_pledged', 'usd_pledged_real', 'usd_goal_real']
    
    big_list=[]
    for object in obj_list:
        some_list=[]
        some_list.extend([
            object.ID, object.name, object.category, object.main_category,
            object.currency, object.deadline, object.goal, object.launched,
            object.pledged, object.state, object.backers, object.country,
            object.usd_pledged, object.usd_pledged_real, object.usd_goal_real
        ])
        big_list.append(some_list)
    df_test = pd.DataFrame(data=big_list, columns=cols)
    print(df_test)
    return df_test