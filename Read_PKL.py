# import pandas
# import joblib

# data = joblib.load("MB1_seed0.pkl")
# print(data)
# data.to_csv('dados.txt', sep='\t', index=False)

import joblib

# Load the .pkl file
data_20 = joblib.load('teste.pkl')



# Print the data to verify its contents
print("Data from s20.pkl:")
print(data_20)

#save as txt
data_20.to_csv('teste.txt', sep='\t', index=False)

