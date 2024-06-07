import pandas as pd
import datetime, warnings, scipy 
warnings.filterwarnings("ignore")

df = pd.read_csv('flightdata.csv')
print(df.head())
print(df.shape)
print(df.isnull().values.any())
print(df.isnull().sum())
df = df.drop('Unnamed: 25', axis=1)
print(df.isnull().sum())

df = df[["MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK", "ORIGIN", "DEST", "CRS_DEP_TIME", "ARR_DEL15"]]
print(df.isnull().sum())

print(df[df.isnull().values.any(axis=1)].head())



df=df.fillna({'ARR_DEL15':1})
print(df.iloc[177:184])


import math

for index, row in df.iterrows():
    df.loc[index, 'CRS_DEP_TIME'] = math.floor(row['CRS_DEP_TIME'] / 100)
print(df.head())




df = pd.get_dummies(df, columns=['ORIGIN', 'DEST'])
print(df.head())


from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(df.drop('ARR_DEL15', axis=1), df['ARR_DEL15'], test_size=0.2, random_state=42)

print(train_x.shape)



print(test_x.shape)

print(train_y.shape)

print(test_y.shape)




from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=13)
model.fit(train_x, train_y)


predicted = model.predict(test_x)
model.score(test_x, test_y)

from sklearn.metrics import roc_auc_score
probabilities = model.predict_proba(test_x)

print(roc_auc_score(test_y,probabilities[:,1]))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(test_y, predicted))

from sklearn.metrics import precision_score

train_predictions = model.predict(train_x)
print(precision_score(train_y, train_predictions))

from sklearn.metrics import recall_score

print(recall_score(train_y, train_predictions))

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


from sklearn.metrics import roc_curve



def predict_delay(departure_date_time, origin, destination):
    from datetime import datetime

    try:
        departure_date_time_parsed = datetime.strptime(departure_date_time, '%d/%m/%Y %H:%M:%S')
    except ValueError as e:
        return 'Error parsing date/time - {}'.format(e)

    month = departure_date_time_parsed.month
    day = departure_date_time_parsed.day
    day_of_week = departure_date_time_parsed.isoweekday()
    hour = departure_date_time_parsed.hour

    origin = origin.upper()
    destination = destination.upper()

    input = [{'MONTH': month,
              'DAY_OF_MONTH': day,
              'DAY_OF_WEEK': day_of_week,
              'CRS_DEP_TIME': hour,
              'ORIGIN_ATL': 1 if origin == 'ATL' else 0,
              'ORIGIN_DTW': 1 if origin == 'DTW' else 0,
              'ORIGIN_JFK': 1 if origin == 'JFK' else 0,
              'ORIGIN_MSP': 1 if origin == 'MSP' else 0,
              'ORIGIN_SEA': 1 if origin == 'SEA' else 0,
              'DEST_ATL': 1 if destination == 'ATL' else 0,
              'DEST_DTW': 1 if destination == 'DTW' else 0,
              'DEST_JFK': 1 if destination == 'JFK' else 0,
              'DEST_MSP': 1 if destination == 'MSP' else 0,
              'DEST_SEA': 1 if destination == 'SEA' else 0 }]

    return model.predict_proba(pd.DataFrame(input))[0][0]

print(predict_delay('1/10/2022 21:45:00', 'JFK', 'SEA'))







import numpy as np

labels = ('9am', 'noon', '3pm', '6pm', '9pm')
values = (predict_delay('30/01/2018 9:00:00', 'SEA', 'ATL'),
          predict_delay('30/01/2018 12:00:00', 'SEA', 'ATL'),
          predict_delay('30/01/2018 15:00:00', 'SEA', 'ATL'),
          predict_delay('30/01/2018 18:00:00', 'SEA', 'ATL'),
          predict_delay('30/01/2018 21:00:00', 'SEA', 'ATL'))
alabels = np.arange(len(labels))
print(values)
plt.bar(alabels, values, align='center', alpha=0.5)
plt.xticks(alabels, labels)
plt.ylabel('Probability of On-Time Arrival')
plt.ylim((0.0, 1.0))
plt.show()











