import pandas as pd
import glob
import os

from sklearn.preprocessing import StandardScaler

"""
Process and standardize data filled with missing values
"""
def Normalization():
    # Step -> combine data and Normalization then train and prediction data
    economic_column = ["Price", "Open", "High", "Low"]
    # Find the fillna data
    files = glob.glob('Data\\fillna\\*.csv')
    time_index = pd.date_range("2015-01-01", "2021-12-31", freq='D')
    df = pd.DataFrame(index = time_index)

    for file in files:
        file_name = os.path.basename(file)
        file_name.replace(".csv", "")
        file_name = file_name.split("_")[0]
        file_df = pd.read_csv(file)
        for i in file_df.columns:
            if i in economic_column:
                print(file_df[i])
                df.loc[:,f'{i}_{file_name}'] = file_df[i].values
                # df[f'{i}_{file_name}'] = file_df[i]
                
    print(df)
    # Choice you y for prediction
    economi_project = ['brent-daily', 'dubai-crude', 'Gold-Price', 'wti-daily']

    # Standardize every Economic Project
    for economic_name in economi_project:
        # X is the field other than the economic data you selected
        X = df.drop(f'Price_{economic_name}', axis=1)
        y = pd.DataFrame(data=df[f'Price_{economic_name}'], index=df.index, columns=[f'Price_{economic_name}'])

        # Normalization
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y)

        # Transform to dataframe
        X_scaled = pd.DataFrame(data=X_scaled, index=X.index, columns=X.columns)
        y_scaled = pd.DataFrame(data=y_scaled, index=y.index, columns=y.columns)
        # Combine the data
        # Save the data to csv  (Our y will all be last column)
        df_scaled = pd.concat([X_scaled, y_scaled], axis=1)

        # Create a directory
        if not os.path.exists("Data\\Normalization"):
            os.makedirs("Data\\Normalization")
        else:
            print("Directory already exists we will not create")
        # Let the index have a name
        df_scaled = df_scaled.rename_axis('Date')
        df_scaled.to_csv(f"Data\\Normalization\\{economic_name}_NormalizationData.csv", header=True)

    # The 0 in the data is because there is a missing position
    # when the missing value was filled in before


def inverse_Normalization():
    """
    Destandardize y_test and y_pred
    """


    
if __name__ == "__main__":
    Normalization()
    



    