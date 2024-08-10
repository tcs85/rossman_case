import pickle
import inflection
import pandas as pd 
import numpy as np
import math
import datetime

# Create Class: need to save the transformation in a pickle to apply the same transformation to new data incoming

class rossmann(object):
    def __init__(self):
        self.home_path='/home/thiago_souza/project_cds/rossmann_case/'
        # Load the RobustScaler for 'competition_distance'
        self.robust_scaler_competition_distance    = pickle.load(open(self.home_path +'parameter/robust_scaler_competition_distance.pkl', 'rb'))
    
        # Load the RobustScaler for 'competition_time_month'
        self.robust_scaler_competition_time_month  = pickle.load(open(self.home_path +'parameter/robust_scaler_competition_time_month.pkl', 'rb'))

        # Load the MinMaxScaler for 'promo_time_week'
        self.minmax_scaler_promo_time_week         = pickle.load(open(self.home_path +'parameter/minmax_scaler_promo_time_week.pkl', 'rb'))
        
        # Load the MinMaxScaler for 'year'
        self.minmax_scaler_year                    = pickle.load(open(self.home_path +'parameter/minmax_scaler_year.pkl', 'rb'))

        # Load the LabelEncoder for 'store_type'
        self.label_encoder_store_type              = pickle.load(open(self.home_path +'parameter/label_encoder_store_type.pkl', 'rb'))
    
    def data_cleanning(self, df1):
        # List of original column names removed columns customer and sales
        cols_old = ['Store', 'DayOfWeek', 'Date', 'Open',
                    'Promo', 'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment',
                    'CompetitionDistance', 'CompetitionOpenSinceMonth',
                    'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek',
                    'Promo2SinceYear', 'PromoInterval']

        # Lambda function to convert camel case to snake case
        snakecase = lambda x: inflection.underscore( x )

        # Apply the function to each column name in cols_old
        cols_new = list( map( snakecase, cols_old ) )

        # Rename the columns of the DataFrame
        df1.columns = cols_new
        
        # Convert the 'date' column in df1 to datetime format
        df1['date']=pd.to_datetime(df1['date'])
        
        #competition_distance
        df1['competition_distance'] = df1['competition_distance'].apply(
            lambda x:200000.0 if math.isnan( x ) else x )
        #Business Context: The chosen value (200000.0 in this case) is an assumptions. If the competition distance is unknown, is reasonable to assume it is very high, indicating no nearby competition.

        # Replace NaN values in 'competition_open_since_month' with the month from 'date'
        df1['competition_open_since_month'] = df1.apply(
            lambda x: x['date'].month if math.isnan(x['competition_open_since_month']) else x['competition_open_since_month'], axis=1)
        #Replacing NaN values in the competition_open_since_month column with the month extracted from the date column is a pragmatic solution to ensure data completeness, maintain temporal consistency, and avoid biases, ultimately leading to more reliable and insightful business analyses and decisions.

        # Replace NaN values in 'competition_open_since_year' with the year from 'date'
        df1['competition_open_since_year'] = df1.apply(
            lambda x: x['date'].year if math.isnan(x['competition_open_since_year']) else x['competition_open_since_year'], axis=1)

        # Replace NaN values in 'promo2_since_week' with the week from 'date'
        df1['promo2_since_week'] = df1.apply(
            lambda x: x['date'].week if math.isnan(x['promo2_since_week']) else x['promo2_since_week'], axis=1)

        # Replace NaN values in 'promo2_since_year' with the year from 'date'
        df1['promo2_since_year'] = df1.apply(
            lambda x: x['date'].year if math.isnan(x['promo2_since_year']) else x['promo2_since_year'], axis=1)

        # Handle 'promo_interval' and create 'is_promo' flag
        month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                    7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
        # df1 = df1.copy()  # Ensure we are working with a copy of the original DataFrame
        df1['promo_interval']=df1['promo_interval'].fillna(0)
        df1['month_map']=df1['date'].dt.month.map(month_map)
        df1['is_promo']=df1.apply(
            lambda x: 0 if x['promo_interval'] == 0 else 1 if x['month_map'] in x['promo_interval'].split(',') else 0, axis=1)

        # Convert 'competition_open_since_month' column to integer type
        df1['competition_open_since_month']=df1['competition_open_since_month'].astype(int)

        # Convert 'competition_open_since_year' column to integer type
        df1['competition_open_since_year']=df1['competition_open_since_year'].astype(int)

        # Convert 'promo2_since_week' column to integer type
        df1['promo2_since_week']=df1['promo2_since_week'].astype(int)

        # Convert 'promo2_since_year' column to integer type
        df1['promo2_since_year']=df1['promo2_since_year'].astype(int)
        
        return df1
    
    def feature_engineering(sel, df2):

        #Create year column
        df2['year']=df2['date'].dt.year

        #Create month column
        df2['month']=df2['date'].dt.month

        #Create year column
        df2['day']=df2['date'].dt.day

        #Create week of year column
        df2['week_of_year']=df2['date'].dt.isocalendar().week

        #Create year week column
        df2['year_week']=df2['date'].dt.strftime('%Y-%U')

        #Create competition since column
        df2['competition_since']=df2.apply(
                                            lambda x: datetime.datetime(year=x['competition_open_since_year'],
                                                                        month=x['competition_open_since_month'],
                                                                        day=1), axis=1)

        df2['competition_time_month']=((df2['date']-df2['competition_since'])/30).apply(
                                                                                        lambda x: x.days).astype(int)

        #Create promo since column
        df2['promo_since']=df2['promo2_since_year'].astype(str)+'-'+df2['promo2_since_week'].astype(str)

        df2['promo_since']=df2['promo_since'].apply(
                                                    lambda x: datetime.datetime.strptime(x+'-1', '%Y-%W-%w')-datetime.timedelta(days=7))

        #Create promo time week column
        df2['promo_time_week'] = ((df2['date'] - df2['promo_since']) / 7).apply(
                                                                                        lambda x: x.days).astype(int)

        #change assortment classification
        df2['assortment']=df2['assortment'].apply(
                                                    lambda x: 'basic' if x =='a' else 'extra' if x == 'b' else 'extended')

        #change state holiday classification
        df2['state_holiday']=df2['state_holiday'].apply(
        
                                                        lambda x: 'public_holiday' if x== 'a' else 'easter_holiday' if x == 'b' else 'christmas' if x=='c' else 'regular_day')
        # Filter df2 to include only rows where 'open' is not equal to 0 and 'sales' is greater than 0
        #remove sales
        df2=df2[(df2['open'] !=0)]
        
        cols_drop=['open', 'promo_interval', 'month_map']
        df2=df2.drop(cols_drop,axis=1)  
            
        return df2
    
    def data_preparation(self,df5):
    
        # Instantiate the scalers

        # Apply RobustScaler to 'competition_distance' column
        df5['competition_distance'] = self.robust_scaler_competition_distance.fit_transform(df5[['competition_distance']].values)


        # Apply RobustScaler to 'competition_time_month' column
        df5['competition_time_month'] = self.robust_scaler_competition_time_month.fit_transform(df5[['competition_time_month']].values)


        # Apply MinMaxScaler to 'promo_time_week' column
        df5['promo_time_week'] = self.minmax_scaler_promo_time_week.fit_transform(df5[['promo_time_week']].values)

        # Apply MinMaxScaler to 'year' column
        df5['year'] = self.minmax_scaler_year.fit_transform(df5[['year']].values)
        
        # state_holiday - One Hot Encoding
        df5 = pd.get_dummies(df5, prefix=['state_holiday'], columns=['state_holiday'])

        # store_type - Label Encoding
        df5['store_type'] = self.label_encoder_store_type.fit_transform(df5['store_type'])

        # assortment - Ordinal Encoding
        assortment_dict = {'basic': 1, 'extra': 2, 'extended': 3}
        df5['assortment'] = df5['assortment'].map(assortment_dict)    

        # Day of Week Transformation
        # Applying sine and cosine transformations to capture the cyclical nature of the week
        df5['day_of_week_sin'] = df5['day_of_week'].apply(lambda x: np.sin(x * (2. * np.pi / 7)))
        df5['day_of_week_cos'] = df5['day_of_week'].apply(lambda x: np.cos(x * (2. * np.pi / 7)))

        # Month Transformation
        # Applying sine and cosine transformations to capture the yearly seasonality
        df5['month_sin'] = df5['month'].apply(lambda x: np.sin(x * (2. * np.pi / 12)))
        df5['month_cos'] = df5['month'].apply(lambda x: np.cos(x * (2. * np.pi / 12)))

        # Day Transformation
        # Applying sine and cosine transformations to capture the monthly periodicity
        df5['day_sin'] = df5['day'].apply(lambda x: np.sin(x * (2. * np.pi / 30)))
        df5['day_cos'] = df5['day'].apply(lambda x: np.cos(x * (2. * np.pi / 30)))

        # Week of Year Transformation
        # Applying sine and cosine transformations to capture the yearly periodicity
        df5['week_of_year_sin'] = df5['week_of_year'].apply(lambda x: np.sin(x * (2. * np.pi / 52)))
        df5['week_of_year_cos'] = df5['week_of_year'].apply(lambda x: np.cos(x * (2. * np.pi / 52)))

        #manual selected compare with hypostheses analises 
        cols_selected= [
        'store',
        'promo',
        'store_type',
        'assortment',
        'competition_distance',
        'competition_open_since_month',
        'competition_open_since_year',
        'promo2',
        'promo2_since_week',
        'promo2_since_year',
        'competition_time_month',
        'promo_time_week',
        'day_of_week_sin',
        'day_of_week_cos',
        'month_sin',
        'month_cos',
        'day_sin',
        'day_cos',
        'week_of_year_sin',
        'week_of_year_cos']
        
        return df5[cols_selected]
    
    def get_prediction (self, model, original_data, test_data):
        #prediction
        pred=model.predict(test_data)
        #join pred into the original data
        original_data['prediction']=np.expm1(pred)
        
        return original_data.to_json(orient= 'records', date_format= 'iso')