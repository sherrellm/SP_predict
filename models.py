import pandas as pd 
import numpy as np 
from Data_eda import load_quarterly

def load_data():
   df=load_quarterly()
   return df 

def fill_na():
   pass

def cross_val():
   pass


if __name__ == '__main__':
	df = load_data()

