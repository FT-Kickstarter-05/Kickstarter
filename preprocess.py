import pandas as pd
import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

import tensorflow as tf
from sys import prefix


def wrangle(data):
    # Load in the data
    # df = pd.read_csv(data_url)

    #data = pd.read_csv('https://raw.githubusercontent.com/FT-Kickstarter-05/Kickstarter/main/2018_ks_data.csv')

    """# Cleaning and preprocessing """

    unneeded_columns = ['ID', 'name']
    data = data.drop(unneeded_columns, axis=1)

    data['usd pledged'] = data['usd_pledged'].fillna(data['usd_pledged'].mean())

    data = data.drop(data.query("state != 'failed' and state != 'successful'").index, axis=0).reset_index(drop=True)


    """# Feature Engineering and Encoding
    ### Change Date to month and year and delete date
    ### For deadline and launched dates
    """

    data['deadline'].apply(lambda x: x[0:4])

    data['deadline'].apply(lambda x: x[5:7])

    data['deadline_year'] = data['deadline'].apply(lambda x: np.float(x[0:4]))
    data['deadline_month'] = data['deadline'].apply(lambda x: np.float(x[5:7]))

    data['launched_year'] = data['launched'].apply(lambda x: np.float(x[0:4]))
    data['launched_month'] = data['launched'].apply(lambda x: np.float(x[5:7]))

    data = data.drop(['deadline', 'launched'], axis=1)

    """### Binary encode 'state' column"""

    data['state'].apply(lambda x: 1 if x == 'successful' else 0)

    data['state'] = data['state'].apply(lambda x: 1 if x == 'successful' else 0)


    """### OneHotEncode 'category','main_category'
    ### 'currency'	, 'country'
    """

    
    def onehot_encode(df, columns, prefixes):
        df = df.copy()
        for column, prefix in zip(columns, prefixes):
            dummies = pd.get_dummies(df[column], prefix=prefix)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(column, axis=1)
        return df

    data = onehot_encode(
        data,
        ['category', 'main_category', 'currency', 'country'],
        ['cat', 'main_cat', 'curr', 'country']
    )
    """# Our Data is fully numeric
    ### We are ready to split and scale the data
    """

    # Columns from full dataset onehot encoding
    full_cols = ['goal', 'pledged', 'state', 'backers', 'usd_pledged', 
    'usd_pledged_real', 'usd_goal_real', 'deadline_year', 'deadline_month', 
    'launched_year', 'launched_month', 'cat_3D Printing', 'cat_Academic',
    'cat_Accessories', 'cat_Action', 'cat_Animals', 'cat_Animation',
    'cat_Anthologies', 'cat_Apparel', 'cat_Apps', 'cat_Architecture',
    'cat_Art', 'cat_Art Books','cat_Audio', 'cat_Bacon', 'cat_Blues', 
    'cat_Calendars','cat_Camera Equipment', 'cat_Candles', 'cat_Ceramics',
    "cat_Children's Books", 'cat_Childrenswear', 'cat_Chiptune',
    'cat_Civic Design', 'cat_Classical Music', 'cat_Comedy', 'cat_Comic Books',
    'cat_Comics', 'cat_Community Gardens', 'cat_Conceptual Art',
    'cat_Cookbooks', 'cat_Country & Folk', 'cat_Couture', 'cat_Crafts',
    'cat_Crochet', 'cat_DIY', 'cat_DIY Electronics', 'cat_Dance', 'cat_Design',
    'cat_Digital Art', 'cat_Documentary', 'cat_Drama', 'cat_Drinks',
    'cat_Electronic Music', 'cat_Embroidery', 'cat_Events', 'cat_Experimental',
    'cat_Fabrication Tools', 'cat_Faith', 'cat_Family', 'cat_Fantasy',
    "cat_Farmer's Markets",'cat_Farms', 'cat_Fashion', 'cat_Festivals',
    'cat_Fiction', 'cat_Film & Video', 'cat_Fine Art', 'cat_Flight',
    'cat_Food', 'cat_Food Trucks', 'cat_Footwear', 'cat_Gadgets', 'cat_Games',
    'cat_Gaming Hardware', 'cat_Glass', 'cat_Graphic Design',
    'cat_Graphic Novels', 'cat_Hardware', 'cat_Hip-Hop', 'cat_Horror',
    'cat_Illustration', 'cat_Immersive', 'cat_Indie Rock', 'cat_Installations',
    'cat_Interactive Design', 'cat_Jazz', 'cat_Jewelry', 'cat_Journalism',
    'cat_Kids', 'cat_Knitting', 'cat_Latin', 'cat_Letterpress',
    'cat_Literary Journals', 'cat_Literary Spaces', 'cat_Live Games',
    'cat_Makerspaces', 'cat_Metal', 'cat_Mixed Media', 'cat_Mobile Games',
    'cat_Movie Theaters', 'cat_Music', 'cat_Music Videos', 'cat_Musical',
    'cat_Narrative Film', 'cat_Nature', 'cat_Nonfiction', 'cat_Painting', 
    'cat_People', 'cat_Performance Art', 'cat_Performances', 'cat_Periodicals',
    'cat_PetFashion', 'cat_Photo', 'cat_Photobooks', 'cat_Photography',
    'cat_Places', 'cat_Playing Cards', 'cat_Plays', 'cat_Poetry', 'cat_Pop',
    'cat_Pottery', 'cat_Print','cat_Printing', 'cat_Product Design',
    'cat_Public Art', 'cat_Publishing', 'cat_Punk', 'cat_Puzzles',
    'cat_Quilts', 'cat_R&B', 'cat_Radio & Podcasts', 'cat_Ready-to-wear',
    'cat_Residencies', 'cat_Restaurants', 'cat_Robots', 'cat_Rock',
    'cat_Romance', 'cat_Science Fiction', 'cat_Sculpture', 'cat_Shorts',
    'cat_Small Batch', 'cat_Software', 'cat_Sound', 'cat_Space Exploration',
    'cat_Spaces', 'cat_Stationery', 'cat_Tabletop Games', 'cat_Taxidermy',
    'cat_Technology', 'cat_Television', 'cat_Textiles', 'cat_Theater',
    'cat_Thrillers', 'cat_Translations', 'cat_Typography', 'cat_Vegan',
    'cat_Video', 'cat_Video Art', 'cat_Video Games', 'cat_Wearables',
    'cat_Weaving', 'cat_Web', 'cat_Webcomics', 'cat_Webseries',
    'cat_Woodworking', 'cat_Workshops', 'cat_World Music', 'cat_Young Adult',
    'cat_Zines', 'main_cat_Art', 'main_cat_Comics', 'main_cat_Crafts',
    'main_cat_Dance', 'main_cat_Design', 'main_cat_Fashion',
    'main_cat_Film & Video', 'main_cat_Food', 'main_cat_Games',
    'main_cat_Journalism', 'main_cat_Music', 'main_cat_Photography',
    'main_cat_Publishing', 'main_cat_Technology', 'main_cat_Theater',
    'curr_AUD', 'curr_CAD', 'curr_CHF', 'curr_DKK', 'curr_EUR', 'curr_GBP',
    'curr_HKD', 'curr_JPY', 'curr_MXN', 'curr_NOK', 'curr_NZD', 'curr_SEK',
    'curr_SGD', 'curr_USD', 'country_AT', 'country_AU', 'country_BE',
    'country_CA', 'country_CH', 'country_DE', 'country_DK', 'country_ES',
    'country_FR', 'country_GB', 'country_HK', 'country_IE', 'country_IT', 
    'country_JP', 'country_LU', 'country_MX', 'country_N,0"', 'country_NL',
    'country_NO', 'country_NZ', 'country_SE', 'country_SG', 'country_US']

    # Create empty dataframe using the full_cols as columns
    df = pd.DataFrame(columns=full_cols)
    
    # fill in the dataframe with our small testing dataframe stored in the data
    # variable
    data_dict = data.to_dict(orient='records')
    df = df.append(data_dict, ignore_index=True)

    # Replace null values with 0
    df.fillna(0, inplace=True)

    return df


def test_func(data):
    # data = wrangle()

    y = data.loc[:, 'state']
    X = data.drop('state', axis=1)

    """### We only want to scale X, we don’t want to scale what we are trying to predict."""

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    """### Each column has a mean of 0, and a variance of 1"""

    pd.DataFrame(X)

    """### Split Data"""

    # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=34)

    """### Modeling and Training"""

    """### How skewed our our classes"""

    """### 40% positive and 60% negative
    ### We will use class_weight from SKLearn
    ### It allows us to generate weights for all the imbalanced classes
    """

    # This will compute class weights on y_train for classes in y_train.unique()
    # with a mode of balanced.
    # class_weights = class_weight.compute_class_weight(
    #                                         "balanced",
    #                                         y_train.unique(),
    #                                         y_train                                                    
    #                                     )
    # class_weights = dict(enumerate(class_weights))

    # inputs = tf.keras.Input(shape=(221,))
    # x = tf.keras.layers.Dense(64, activation='relu')(inputs)
    # x = tf.keras.layers.Dense(64, activation='relu')(x)
    # outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    # model = tf.keras.Model(inputs, outputs)


    # model.compile(
    #     optimizer='adam',
    #     loss='binary_crossentropy',
    #     metrics=[
    #         'accuracy',
    #         tf.keras.metrics.AUC(name='auc') # Area Under the Curve of the (ROC)Receiver Operating Curve
    #     ]
    # )
    # batch_size = 64
    # epochs = 100 # We are using a large amount of epochs because we are using early stopping

    # print('fitting model')

    # history = model.fit(
    #     X_train,
    #     y_train,
    #     validation_split=0.2,
    #     class_weight=class_weights, # This will assighn weights to the two classes 
    #     #to penalize the out of balance 
    #     batch_size=batch_size,
    #     epochs=epochs,
    #     callbacks=[
    #         tf.keras.callbacks.EarlyStopping( # This is a great callback that lets us monitor a metric or loss value
    #             monitor='val_loss',
    #             patience=3,
    #             restore_best_weights=True,
    #             verbose=1
    #         )
    #     ],
    #     verbose=2
    # )

    print('model fit')

    # pickle model
    #filename = 'kickstarter.sav'

    #print('saving model')
    #model.save('saved_model')

    print('loading model')
    new_model = tf.keras.models.load_model('saved_model')
    #new_model.summary()
    # pred_len = 5
    predictions = new_model.predict(X)
    pred_out = ['failed' if val < 0.25 else 'successful' for val in predictions]
    return pred_out
#print(test[:5])


"""# Results"""
#model_test = pickle.load(open(filename, 'rb'))

#print('evaluating model')

#new_model.evaluate(X_test, y_test)

# data_url = 'https://raw.githubusercontent.com/FT-Kickstarter-05/Kickstarter/main/2018_ks_data.csv'
# df = wrangle(data_url)
