import pandas as pd 
import numpy as np 
from Data_eda import load_quarterly

def load_data():
   df=load_quarterly()
   return df 

def fill_na():
   pass

def cross_val(X,y):


   pass


if __name__ == '__main__':
	quarterly = load_data()
   changes = load_changes()
   df = join_dfs(quartely, changes)

