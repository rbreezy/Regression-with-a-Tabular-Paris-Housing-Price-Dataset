import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

data = pd.read_csv('train.csv')

# min-max scaler
min_max_col_to_scaled = ['numberOfRooms','cityPartRange','numPrevOwners','hasGuestRoom']
scaler = MinMaxScaler()
data[min_max_col_to_scaled] = scaler.fit_transform(data[min_max_col_to_scaled])

# z-score normalization
z_score_cols_to_scale = ['garage','floors']
scaler = StandardScaler()
data[z_score_cols_to_scale] = scaler.fit_transform(data[z_score_cols_to_scale])

# making X and y
X = data[['numberOfRooms','hasYard','hasPool','floors','cityPartRange',
                  'numPrevOwners','isNewBuilt','hasStormProtector','hasStorageRoom',
                  'hasGuestRoom','garage']]
y = data['price']

#split the data into train and val
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

