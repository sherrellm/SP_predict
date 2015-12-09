import pandas as pd 
import numpy as np 
from Quandl import Quandl

auth_tok = 

def get_stock(ticker):
	df = Quandl.get("SF1/{}".format(ticker), authtoken=auth_tok, trim_start="2007-12-30", returns="pandas")
	print df.info()
	return df 

if __name__ == '__main__':
	df = get_stock("GOOGL_MARKETCAP")