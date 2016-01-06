import pandas as pd 
import numpy as np 
from Data_eda import create_SP_500_member_df, load_changes
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import cPickle as pickle

def load_data():
   '''Loads the data frame 
      Input:None
      Output: DataFrame of financial data and S&P 500 Membership ready for modeling
   '''
   df = create_SP_500_member_df()
   return df 


def cross_val(df, model_name):
   ''' Sets up cross validation where for each quarter the prediction is based off the previous' quarter data
   Input: df: the dataframe to run hte model on, model_name:the sklean function to model_name
   Output: The sklearn model object of the last quarter and prints the cross validation metrics
   '''
   df = df.reset_index(drop=True)
   changes=load_changes()
   temp_df =  pd.DataFrame()
   temp_df['quarters_list'] =[row.quarter.ordinal for i,row in df.iterrows()]
   temp =[row.Quarter_removed.ordinal for i,row in changes.iterrows()]

   y_pred_folds = np.array([])
   y_test_folds = np.array([])
   loops = 0 
   for period in xrange(np.min(temp), np.max(temp)-1):
      X_prev, y_prev, tickers_prev = split_data(df[temp_df['quarters_list'] == period-1])
      X_cur, y_cur, tickers_cur = split_data(df[temp_df['quarters_list'] == period])
      X_fut, y_fut, tickers_fut = split_data(df[temp_df['quarters_list'] == period+1])
      y_train, y_test, tickers = calculate_changes(y_prev,y_cur,y_fut,tickers_prev,tickers_cur,tickers_fut)
      X_train = X_cur
      X_test = X_fut
      model = model_name(X_train,X_test,y_train,y_train)
      model.fit(X_train,y_train)
      print model.predict(X_test).shape
      y_pred_folds = np.append(y_pred_folds, model.predict(X_test))
      y_test_folds = np.append(y_test_folds, y_train)

   cross_val_score(y_pred_folds,y_test_folds,"{}".format(model_name))

   return model 

def calculate_changes(y1,y2,y3,tick1,tick2,tick3):
   '''
   Input: The labels and the tickers for the prevoius, current and future quarters and insures all the shapes are the same
   Output: The training labels, testing labels and the associated tickers
   '''
   #represent all tickers in the current and prevoius quarter so y_test and y_train have the same shape
   tickers = set(tick2)
   # tickers.update(tick2)
   tickers.intersection(tick3)
   tickers = list(tickers)
   y_train = np.zeros(len(tickers)).flatten().tolist()
   y_test = np.zeros(len(tickers)).flatten().tolist()
   for i,val in enumerate(y1):
      try:
         y_train[tickers.index(tick1[i])] = val 
         break
      except ValueError:
         pass
      
   
   for i,val in enumerate(y2):
      if y_train[tickers.index(tick2[i])] == 1  and val == 0:
         try:
            y_train[tickers.index(tick2[i])] == 1
            break
         except ValueError:
            pass
      else:
         try:
            y_train[tickers.index(tick2[i])] == 0
            break
         except ValueError:
            pass  

      try:
         y_test[tickers.index(tick2[i])] == val
         break
      except ValueError:
         pass           
  
   for i,val in enumerate(y3):
      if y_test[tickers.index(tick3[i])] == 1  and val == 0:
         try:
            y_train[tickers.index(tick3[i])] == 1
            break
         except ValueError:
            pass         
      else:
         try:
            y_train[tickers.index(tick3[i])] == 0
            break
         except ValueError:
            pass  
   return y_train, y_test, tickers 

def split_data(df):
   ''' Splits the data into features and labels
   Input: DataFrame to model on 
   Output: X: features, y:labels
   '''

   df['quarter'] = [row.quarter.ordinal for i,row in df.iterrows()]
   tickers = df.pop('ticker').values
   y = df.pop('SP_500_member').values
   X = df.values 
   return X, y, tickers

def cross_val_score(y_pred,y_test,model_name=None):
   ''' Scores the cross validated predictions
   Input: The predicted and true labels, optional the model name 
   Output: Prints out TPR,TNR, AUC, and the confusion Matrix
   '''

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
   ''' 
   Method for calling sklearn's RandomForestClassifier
   Input: X_train: training features,X_test: test features, y_train: training labels,y_test:Test labels
   Output: sklearn's RandomForestClassifier object
   '''
   model = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
   return model


def gradient_boost(X_train,X_test,y_train,y_test):
   ''' 
   Method for calling sklearn's Gradient Boosted Treess Classifer
   Input: X_train: training features,X_test: test features, y_train: training labels,y_test:Test labels
   Output: sklearn's Gradient Boosted Treess Classifer object
   '''
   model = GradientBoostingClassifier(learning_rate=0.001, n_estimators=1000)
   return model 

def load_model_pickle(filename):
   '''
   Loads the pickled model 
   Input: the filname to save the pickle as 
   Output: writes the pickle to disk in the current directory
   '''
   with open("{}.pkl".format(filename)) as f:
      model_unpickled = pickle.load(f)
      return model_unpickled

def pickle_model(model,model_name):
   '''
   Writes the pickled model to disk
   Input: the model object to pickle and the filename to save the pickle as 
   Output: writes the pickle to disk in the current directory
   '''

   with open("{}.pkl".format(model_name), 'w') as f:
      pickle.dump(model, f)


def main():
   '''
   Main function for running the models from cmd line 
   '''
   data = load_data()
   #fill NaN's as sufficently negative numbers that they are out bounds
   data = data.fillna(-1e20)
   model = cross_val(data, random_forest)
   # pickle_model(model,random_forest)
   # model = cross_val(data, gradient_boost)
   # pickle_model(model,gradient_boost)
   # print df
   return data, model


if __name__ == '__main__':
   df, model = main()





