import pandas as pd

from sklearn.linear_model import Lasso

data = pd.read_csv('train.csv')
print('the data shape is : ', data.shape)

X = data.drop('price', axis=1)
y = data['price']

feature_selection = Lasso(0.1)
feature_selection.fit(X,y)

print('the feature importance is : ')
print(feature_selection.coef_)

for i in range(len(feature_selection.coef_)):
    print(data.columns[i],feature_selection.coef_[i])


"""
id -2.7199390604301525
squareMeters 34.301006933712834
numberOfRooms 6688.7177781976225
hasYard 8943.827495736998
hasPool 43516.008557324094
floors 1675.735034118779
cityCode 1.1312022774898123
cityPartRange -12482.778151702087
numPrevOwners -11076.390507243252
made 367.36932684671683
isNewBuilt 19365.69832372491
hasStormProtector 61602.445567644885
basement -25.96858567745619
attic -5.430477609040865
garage -946.6246313931138
hasStorageRoom 7060.643332174227
hasGuestRoom -2971.740612423577
"""
