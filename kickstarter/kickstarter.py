from .models import DB, Campaign


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