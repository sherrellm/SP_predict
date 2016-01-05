import pandas as pd 
import numpy as np 
from Data_eda import create_SP_500_member_df, load_changes

def load_data():
   df = create_SP_500_member_df()
   return df 

def fill_na(df,val):
   return df.fill_na(val=val)

def cross_val(df, model_name):
   df = df.reset_index(drop=True)
   changes=load_changes()
   for period in xrange(changes.Quarter_removed.min().ordinal, changes.Quarter_removed.max().ordinal-1):
      X_train, y_1 =  split_data(df[df.quarter == period])
      X_test, y2 =  split_data(df[df.quarter == period+1])
      print X_train.shape
      model = model_name(X_train,X_test,y_train,y_test)


   return model 

def split_data(df):
   y = df.pop('SP_500_member').values
   X = df.values 
   return X, y 

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
   model = RandomForestClassifier(n_estimators=2000, n_jobs=-1)
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
   model = cross_val(data, random_forest)
   changes=load_changes()
   print changes.Quarter_removed.min().ordinal

   return data 


if __name__ == '__main__':
   df = main()

	



