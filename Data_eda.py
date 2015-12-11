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
	df['ticker'] = df['ticker_metric'].apply(lambda x: x.split('_',1)[0])
	df['metric'] = df['ticker_metric'].apply(lambda x: x.split('_',1)[1])
	df=df.drop('ticker_metric', axis=1)
	df=pd.pivot_table(df,index=['date','ticker'], columns='metric',values='value')
	return df

def save_pivot(df):
	df.to_csv("data/pivot.csv")

def read_pivot():
	df = pd.read_csv("data/pivot.csv")
	return df 

def generate_quarterly(df):
	df = read_pivot()
	df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
	columns = ['date', 'ticker', 'ACCOCI_ARQ',
   'ASSETSAVG_ART', 'ASSETSC_ARQ','ASSETSNC_ARQ', 'ASSETS_ARQ',
   'ASSETTURNOVER_ART', 'BVPS_ARQ','CAPEX_ARQ', 
   'CASHNEQ_ARQ', 'COR_ARQ', 'CURRENTRATIO_ARQ','DEBT_ARQ', 
   'DEPAMOR_ARQ','DE_ARQ', 'DILUTIONRATIO_ARQ',
   'DIVYIELD', 'DPS_ARQ','EBITDAMARGIN_ART', 'EBITDA_ARQ',
   'EBIT_ARQ','EBT_ARQ','EPSDILGROWTH1YR_ART',  'EPSDIL_ARQ', 
   'EPSGROWTH1YR_ART',  'EPS_ARQ', 'EQUITYAVG_ART',  'EQUITY_ARQ', 'EV',
   'EVEBITDA_ART',  'EVEBIT_ART',  'EVENT','FCFPS_ARQ', 'FCF_ARQ',  
   'FILINGDATE', 'FILINGTYPE', 'GP_ARQ','GROSSMARGIN_ART',
   'INTANGIBLES_ARQ', 'INTERESTBURDEN_ART','INTEXP_ARQ', 
   'INVCAPAVG_ART','INVCAP_ARQ', 'INVENTORY_ARQ','LEVERAGERATIO_ART', 
   'LIABILITIESC_ARQ', 'LIABILITIESNC_ARQ', 'LIABILITIES_ARQ',
   'MARKETCAP', 'NCFCOMMON_ARQ', 'NCFDEBT_ARQ','NCFDIV_ARQ', 
   'NCFF_ARQ','NCFI_ARQ','NCFOGROWTH1YR_ART',  'NCFO_ARQ', 'NCFX_ARQ',
   'NCF_ARQ','NETINCCMN_ARQ','NETINCDIS_ARQ','NETINCGROWTH1YR_ART', 
   'NETINC_ARQ','NETMARGIN_ART', 'PAYABLES_ARQ','PAYOUTRATIO_ART', 'PB_ARQ',
   'PE1_ART','PE_ART', 'PREFDIVIS_ARQ',
   'PRICE', 'PS1_ART',  'PS_ART', 'RECEIVABLES_ARQ',  'RETEARN_ARQ', 
   'REVENUEGROWTH1YR_ART', 'REVENUE_ARQ','RND_ARQ','ROA_ART', 'ROE_ART', 
   'ROIC_ART',  'ROS_ART',  'SGNA_ARQ',
   'SHAREFACTOR', 'SHARESBAS', 'SHARESWADIL_ARQ', 'SHARESWAGROWTH1YR_ART', 
   'SHARESWA_ARQ',  'SPS_ART','TANGIBLES_ARQ', 'TAXEFFICIENCY_ART',
   'TAXEXP_ARQ','TBVPS_ARQ', 'WORKINGCAPITAL_ARQ' ]
	
	df = df[columns]
	df = df.set_index('date')
	df['quarter'] = df.index.to_period('Q')
	df.groupby(['quarter', 'ticker']).mean()
	df.to_csv("data/quarterly.csv")

def load_quarterly():
	df = pd.read_csv("data/quarterly.csv")
	return df 


if __name__ == '__main__':
	df = load_database()
	df = process_database(df)
	save_pivot(df)
	generate_quarterly(df)