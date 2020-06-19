import pandas as pd


train_df= pd.read_csv('train.csv') # imports data
X_train= train_df.drop("Survived",axis=1)
Y_train= train_df["Survived"]


train_df['Sex'] = train_df['Sex'].map({'female': 1, 'male': 0}).astype(int) # Training on Sex column

# Calculating the correlation
print("Correlation: ",train_df['Survived'].corr(train_df['Sex'])*100)
print("Women and Children left Titanic the first and the data just proved that there is a correlation.")
print("Correlation is pretty High more than 50% so we cannot drop Sex")
