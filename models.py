import pandas as pd 
import numpy as np 
from Quandl_eda import read_pivot

def load_data():
   df=load_quarterly()
   return df 

if __name__ == '__main__':
	df = load_data()

