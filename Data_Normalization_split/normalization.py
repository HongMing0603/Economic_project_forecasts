import pandas as pd
import glob
import os

from sklearn.preprocessing import StandardScaler

"""
Process and standardize data filled with missing values
"""
# Step -> combine data and Normalization then train and prediction data
economi_column = ["Price", "Open", "High", "Low"]
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
        if i in economi_column:
            print(file_df[i])
            df.loc[:,f'{i}_{file_name}'] = file_df[i].values
            # df[f'{i}_{file_name}'] = file_df[i]
            
print(df)
# Choice you y for prediction
economi_project = ['1.brent-daily', '2.dubai-crude', '3.Gold-Price', '4.wti-daily']
economi_choice = ""
print(economi_project)
economi_choice = input("Please select economi project you want to forecasting(Enter a number):")
# Judge the option selected
# Your economi_choice should be like "....-....",Insted of ....._.....
# Because the code behind will report an error
if economi_choice =="1":
    print("You choice brent-daily")
    economi_choice = 'brent-daily'
elif economi_choice == "2":
    print("You choice dubai-crude")
    economi_choice = 'dubai'
elif economi_choice == "3":
    print("You choice Gold-Price")
    economi_choice = 'Gold-Price'
elif economi_choice == "4":
    print("You choice wti-daily")
    economi_choice = 'wti-daily'
else:
    print("You set a error code!!")

# X is the field other than the economic data you selected
X = df.drop(f'Price_{economi_choice}', axis=1)
y = pd.DataFrame(data=df[f'Price_{economi_choice}'], index=df.index, columns=[f'Price_{economi_choice}'])

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
df_scaled.to_csv("Data\\Normalization\\NormalizationData.csv", header=True)



    
    



    