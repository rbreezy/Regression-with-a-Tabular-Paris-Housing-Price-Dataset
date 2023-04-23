import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle

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

# model building
model = Sequential()
model.add(Dense(35, input_shape=(11,), activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='linear'))

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

model.compile(loss=root_mean_squared_error, optimizer='adam')
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=80, batch_size=85, verbose=1)

# save the model as a .pkl file
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)