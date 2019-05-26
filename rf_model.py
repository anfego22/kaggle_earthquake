import pymongo
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

conection = pymongo.MongoClient()
db = conection.earthquake
cursor = db.augmented_train.find()
augmented_data = [el for el in cursor]


augmented_data = pd.DataFrame(augmented_data)
X = augmented_data[['max', 'mean', 'min', 'sd']].values
y = augmented_data['time_to_failure'].values
scaler = StandardScaler()
svm = NuSVR()
rf = RandomForestRegressor(n_estimators=120, n_jobs=-1,
                           min_samples_leaf=1,
                           max_features = "auto", max_depth=15)
kf = KFold(5)
fold_ = 0
for trn_, val_ in kf.split(X):
    print("Current Fold: {}".format(fold_))
    trn_x, trn_y = X[trn_], y[trn_]
    val_x, val_y = X[val_], y[val_]
    trn_x = scaler.fit_transform(trn_x)
    svm.fit(trn_x, trn_y)
    rf.fit(trn_x, trn_y)
    val_x = scaler.fit_transform(val_x)
    svm_pred = svm.predict(val_x)
    rf_pred = rf.predict(val_x)
    print('SVM MAE: {}'.format(mean_absolute_error(val_y, svm_pred)))
    print('RF MAE: {}'.format(mean_absolute_error(val_y, rf_pred)))
    print('RF-SVM MAE: {}'.format(mean_absolute_error(
        val_y, 0.8*rf_pred + 0.2*val_pred)))
    fold_ += 1

submission = []
scaler = StandardScaler()
for f in os.listdir(PATH):
    res = {}
    data = pd.read_csv(PATH + f, sep=',')
    data['acoustic_data'] = scaler.fit_transform(data['acoustic_data'].\
                                                 values.reshape(-1, 1))
    features = data[data.columns[data.columns != 'time_to_failure']].values
    features = features[~np.isnan(features).any(axis=1)]
    features = np.expand_dims(features, axis=0)
    res['seg_id'] = f[:-4]
    res['time_to_failure'] = model.predict(features)[0][0]
    submission.append(res)
