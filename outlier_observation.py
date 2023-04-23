import pandas as pd
import seaborn as sns

data = pd.read_csv('train.csv')

features_using = ['numberOfRooms','hasYard','hasPool','floors','cityPartRange',
                  'numPrevOwners','isNewBuilt','hasStormProtector','hasStorageRoom',
                  'hasGuestRoom','garage','squareMeters','cityCode','made','basement',
                  'attic']

for feature in features_using:
    print(data[feature].describe())

# after the above we found out the following coloumn with outlier it need z-score normalization
"""
garage
floors
basement
made
"""

# min-max 
"""
numberOfRooms
cityPartRange
numPrevOwners
hasGuestRoom
"""
