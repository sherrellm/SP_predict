import pandas as pd 
import numpy as np 
from Data_eda import load_quarterly

def load_data():
   df=load_quarterly()
   return df 

if __name__ == '__main__':
	df = load_data()

