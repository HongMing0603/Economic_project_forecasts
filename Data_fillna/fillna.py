import pandas as pd
import glob
import os

# fillna Package
from sklearn.impute import SimpleImputer


files = glob.glob('Data\RangeChanges\*.csv')

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
    