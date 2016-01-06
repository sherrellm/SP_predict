import pandas as pd 
import numpy as np 
from Data_eda import create_SP_500_member_df, load_changes
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import cPickle as pickle

def load_data():
   df = create_SP_500_member_df()
   return df 


def cross_val(df, model_name):
   df = df.reset_index(drop=True)
   changes=load_changes()
   temp_df =  pd.DataFrame()
   temp_df['quarters_list'] =[row.quarter.ordinal for i,row in df.iterrows()]
   for period in xrange(changes.Quarter_removed.min().ordinal, changes.Quarter_removed.max().ordinal-1):
      X_train, y_train, tickers_train =  split_data(df[temp_df['quarters_list'] == period])
      X_test, y_test, tickers_test =  split_data(df[temp_df['quarters_list'] == period+1])
      model = model_name(X_train,X_test,y_train,y_train)

   return model 

def split_data(df):

   df['quarter'] = [row.quarter.ordinal for i,row in df.iterrows()]
   tickers = df.pop('ticker').values

   y = df.pop('SP_500_member').values
   X = df.values 
   return X, y , tickers 

def cross_val_score(y_pred,y_test,model_name=None):
   tp = np.count_nonzero(np.where(np.logical_and(y_pred== 1, y_test== 1)))
   tn = np.count_nonzero(np.where(np.logical_and(y_pred== 0, y_test== 0)))
   fp = np.count_nonzero(np.where(np.logical_and(y_pred== 1, y_test== 0)))
   fn = np.count_nonzero(np.where(np.logical_and(y_pred== 0, y_test== 1)))
   if model_name:
      print model_name
   print "TPR: {}".format(tp/float(tp+fn))
   print "TNR: {}".format(tn/float(fp+tn))
   print "Percision: {}".format(tp/float(tp+fp))
   print "AUC: {}".format(roc_auc_score(y_test,y_pred))
   print "Confusion Matrix"
   print tp,fp
   print fn,tn 

def random_forest(X_train,X_test,y_train,y_test):
   model = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
   model.fit(X_train,y_train)
   y_pred = model.predict(X_test)
   cross_val_score(y_pred,y_test,"Random Forest")
   return model


def gradient_boost(X_train,X_test,y_train,y_test):
   model = GradientBoostingClassifier(learning_rate=0.001, n_estimators=1000, subsample=.7)
   model.fit(X_train,y_train)
   y_pred = model.predict(X_test)
   cross_val_score(y_pred,y_test, "Gradient Boost")
   return model 

def load_model_pickle(filename):
   with open("{}.pkl".format(filename)) as f:
      model_unpickled = pickle.load(f)
      return model_unpickled

def pickle_model(model,model_name):
   with open("{}.pkl".format(model_name), 'w') as f:
      pickle.dump(model, f)


def main():
   data = load_data()
   #fill NaN's as sufficently negative numbers that they are out bounds
   data = data.fillna(-1e20)
   model = cross_val(data, random_forest)
   # print df


   return data 


if __name__ == '__main__':
   df = main()
   X, y, tickers = split_data(df)
	



