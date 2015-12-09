import pandas as pd 
import numpy as np 
from Quandl import Quandl

auth_tok = None

def get_stock(ticker):
	df = Quandl.get("SF1/{}".format(ticker), authtoken=auth_tok, trim_start="2007-12-30", returns="pandas")
	print df.info()
	return df 

def load_constiuents():
	df=pd.read_csv("data/S&P_comp_20151209")
	return df 

def load_database():
	df=pd.read_csv("data/SF1_20151209.csv")
	df.columns= ['ticker_metric','date','value']
	return df 

def process_database(df):
	print "processing"
	df['ticker'] = df['ticker_metric'].apply(lambda x: x.split('_',1)[0])
	print "1st split"
	df['metric'] = df['ticker_metric'].apply(lambda x: x.split('_',1)[1])
	print "2nd split done "
	df=df.drop('ticker_metric', axis=1)
	print "droped"
	# df['date'] = pd.to_datetime(df['date'],format="%Y-%M-%d")

	# df['date'] = pd.DatetimeIndex(df['date'])
	print "pivot in "
	df=pd.pivot_table(df,index=['date','ticker'], columns='metric',values='value')
	return df
def save_pivot(df):
	df.to_csv("data/pivot.csv")

def read_pivot():
	df = pd.read_csv("data/pivot.csv")
	return df 

if __name__ == '__main__':
	df = load_database()
	df = process_database(df)
	df.info()
	print df.head(50)
	# print df.columns
	save_pivot(df)