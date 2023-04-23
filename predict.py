import pandas as pd
import pickle
from keras import backend as K

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

test_data = pd.read_csv('test.csv')

min_max_col_to_scaled = ['numberOfRooms','cityPartRange','numPrevOwners',
                         'hasGuestRoom','squareMeters','cityCode']

z_score_cols_to_scale = ['garage','floors','basement','made','basement','attic']

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('min_max_scaler.pkl', 'rb') as file:
    min_max_scaler = pickle.load(file)

with open('z_score_scaler.pkl', 'rb') as file:
    z_score_scaler = pickle.load(file)

test_data[min_max_col_to_scaled] = min_max_scaler.transform(test_data[min_max_col_to_scaled])
test_data[z_score_cols_to_scale] = z_score_scaler.transform(test_data[z_score_cols_to_scale])

ids = test_data['id']

test_data = test_data[['numberOfRooms','hasYard','hasPool','floors','cityPartRange',
                  'numPrevOwners','isNewBuilt','hasStormProtector','hasStorageRoom',
                  'hasGuestRoom','garage','squareMeters','cityCode','made','basement',
                  'attic']]

y_pred = model.predict(test_data)
new = []
for i in y_pred:
    new.append(i[0]) 
print(new)   
preds = pd.Series(new)

df = pd.concat([ids, preds], axis=1, keys=['id','price'])
df.to_csv('sub.csv', index=False)
