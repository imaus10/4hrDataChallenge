import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import lightgbm as lgbm

dtypes = { 'Identifier': str,
           'Occurrence Datetime': str,
           'Day of Week': 'category',
           'Occurrence Month': 'category',
           'Occurrence Day': np.uint8,
           'Occurrence Year': np.uint16,
           'Occurrence Hour': np.uint8,
           'CompStat Month': np.uint8,
           'CompStat Day': np.uint8,
           'CompStat Year': np.uint16,
           'Offense': 'category',
           'Sector': 'category',
           'Precinct': 'category',
           'Borough': 'category',
           'Jurisdiction': 'category',
           'XCoordinate': np.int64,
           'YCoordinate': np.int64,
           'Location 1': str,
           'Occurrence Date': str }
train = pd.read_csv('data/NYPD_7_Major_Felony_Incidents_train.csv',
                    dtype=dtypes)
test = pd.read_csv('data/NYPD_7_Major_Felony_Incidents_test.csv', dtype=dtypes)

X = train.append(test)
y = X.pop('Offense')
splitidx = train.shape[0]
catcols = ['Sector','Borough','Jurisdiction']
for c in catcols:
    X[c] = X[c].astype('category')

baseline = pd.DataFrame(['GRAND LARCENY']*test.shape[0])
baseline_score = accuracy_score(baseline, test['Offense'])
print('baseline: %f' % baseline_score)

# make Occurrence Month a number, like CompStat Month
monthmap = {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May': 5, 'Jun': 6,
            'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12}
X['Occurrence Month'] = X['Occurrence Month'].map(monthmap)
# add minute from the Datetime
X['Occurrence Minute'] = pd.to_datetime(X['Occurrence Datetime']).dt.minute
# split the Location 1 string into its values
loc = X['Location 1'].str.slice(1,-1).str.split(',', expand=True).astype(np.float)
X[['Loc1X','Loc1Y']] = loc
X.drop(['Identifier','Occurrence Datetime','Occurrence Date','Location 1'], axis=1, inplace=True)

yLabels = y.cat.categories
y = y.cat.codes
cats = X.select_dtypes(include=['category'])
cat_ids = cats.apply(lambda c: c.cat.codes)
# X.drop(cat_ids.columns, axis=1, inplace=True)
# X[cat_ids.columns] = cat_ids

# use binary y/n columns instead of category IDs
X.drop(cats.columns, axis=1, inplace=True)
one_hot_cats = pd.get_dummies(cats)
X[one_hot_cats.columns] = one_hot_cats

# throw it in a GBM
n_splits=3
skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
score = 0
predictions = []
# the CV is not super useful right now...
# i set it up to do some basic tuning but ran out of time
for train_idx, val_idx in skf.split(X.iloc[:splitidx], y.iloc[:splitidx]):
    print('training...')
    X_val = X.loc[val_idx]
    y_val = y.loc[val_idx]
    d_train = lgbm.Dataset(X.loc[train_idx], label=y.loc[train_idx])
    d_val = lgbm.Dataset(X_val, label=y_val)
    params = { 'application': 'multiclass',
               'metric': 'multi_error',
               'num_classes': len(yLabels) }
    model = lgbm.train(params, train_set=d_train, valid_sets=d_val)
    curr_score = accuracy_score(model.predict(X_val).argmax(axis=1), y_val)
    score += curr_score
    print('val score: %f' % curr_score)
    print('predicting...')
    predictions.append(model.predict(X.iloc[splitidx:]))
print('avg fold score: %f' % (score/n_splits))

# avg the class probabilities
# to get an avg prediction across the 3 folds
avg_prediction = predictions[0]
for i in range(1,n_splits):
    avg_prediction += predictions[i]
avg_prediction /= n_splits
test_pred = avg_prediction.argmax(axis=1)
test_score = accuracy_score(test_pred, y.iloc[splitidx:])
print('GBM test score: %f' % test_score)

# quick copy-paste with RandomForest at the end to see if that does any better...
# from sklearn.ensemble import RandomForestClassifier
# score = 0
# predictions = []
# test_scores = []
# for train_idx, val_idx in skf.split(X.iloc[:splitidx], y.iloc[:splitidx]):
#     clf = RandomForestClassifier()
#     print('training...')
#     clf.fit(X.loc[train_idx], y.loc[train_idx])
#     val_pred = clf.predict(X.loc[val_idx])
#     curr_score = accuracy_score(val_pred, y.loc[val_idx])
#     print('val score: %f' % curr_score)
#     score += curr_score
#     test_pred = clf.predict(X.iloc[splitidx:])
#     predictions.append(test_pred)
#     test_scores
# print('avg fold score: %f' % (score/3))
