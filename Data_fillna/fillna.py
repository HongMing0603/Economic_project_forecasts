import pandas as pd
import glob
import os

# fillna Package
from sklearn.impute import SimpleImputer

# Define the Economic_list
economic_column = ["Price", "Open", "High", "Low"]

files = glob.glob('Data\RangeChanges\*.csv')

def fillna():

    for file in files:
        df = pd.read_csv(file)
        # set the Date column
        df = df.rename(columns={df.columns[0]:'Date'})
        file_name = os.path.basename(file)
        file_name = file_name.replace(".csv", "")
        # set DataFrame index
        df = df.set_index("Date")
        print(f"Now file is {file_name} ")
        print(df.isnull().sum())

        # Use SimpleImputer fillna
        imputer = SimpleImputer(strategy="mean")
        imputed_data = imputer.fit_transform(df)

        # Create a DataFrame
        imputed_df = pd.DataFrame(imputed_data, columns=df.columns, index=df.index)
        print(imputed_df)

        directory_name = "Data/fillna"

        # Create a fillna Directory
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)
        else:
            print("Directory alreay exists")

        # Create to csv
        imputed_df.to_csv(f'{directory_name}/{file_name}_fillnaData.csv', index=True)
        # Check dataFrame number of isnull
        print(imputed_df.isnull().sum().sum())
        
    

# Create a DataFrame for All-fillna files
time_index = pd.date_range("2015-01-01", "2021-12-31", freq='D')
df = pd.DataFrame(index = time_index)

def FillnaCombine():
    """
    Combine for each fillna csv
    Output -> DataFrame = Combine fillna data
    
    """
    # Combine from various CSVS
    files = glob.glob('Data\\fillna\\*.csv')
    # input data into dataframe and rename for column
    for file in files:
        # In here file name is the economic_program's name
        file_name = os.path.basename(file)
        file_name = file_name.split("_")[0]
        # Read the data so that it can fit into ours new DataFrame
        file_df = pd.read_csv(file)
        # Determine if the fields in the csv file are what we want
        # Take out the individual column
        for column in file_df.columns:
            if column in economic_column:
                # put data into ours new dataframe
                df.loc[:, f'{file_name}_{column}'] = file_df[column].values

    # Output this complete DataFrame to csv
    df.index.name = 'Date'
    df.to_csv('Data\\fillna\\Combine\\fillna_Combine.csv', index=True)
    # Return the dataFrame
    return df
