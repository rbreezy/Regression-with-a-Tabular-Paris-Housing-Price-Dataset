import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle

data = pd.read_csv('train.csv')

# min-max scaler
min_max_col_to_scaled = ['numberOfRooms','cityPartRange','numPrevOwners',
                         'hasGuestRoom','squareMeters','cityCode']
min_max_scaler = MinMaxScaler()
data[min_max_col_to_scaled] = min_max_scaler.fit_transform(data[min_max_col_to_scaled])

# z-score normalization
z_score_cols_to_scale = ['garage','floors','basement','made','basement','attic']
z_score_scaler = StandardScaler()
data[z_score_cols_to_scale] = z_score_scaler.fit_transform(data[z_score_cols_to_scale])

data = data[['numberOfRooms','hasYard','hasPool','floors','cityPartRange',
                  'numPrevOwners','isNewBuilt','hasStormProtector','hasStorageRoom',
                  'hasGuestRoom','garage','squareMeters','cityCode','made','basement',
                  'attic','price']]

# making X and y
X = data.drop('price', axis=1)
y = data['price']

#split the data into train and val
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# model building
model = Sequential()
model.add(Dense(60, input_shape=(16,), activation='relu',kernel_regularizer=l2(0.001)))
model.add(Dense(35, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dense(15, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dense(7, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dense(1, activation='linear'))

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

model.compile(loss=root_mean_squared_error, optimizer='adam')
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=90, batch_size=100, verbose=1)

# save the model as a .pkl file
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

with open('min_max_scaler.pkl', 'wb') as f:
    pickle.dump(min_max_scaler, f)

with open('z_score_scaler.pkl', 'wb') as f:
    pickle.dump(z_score_scaler, f)

