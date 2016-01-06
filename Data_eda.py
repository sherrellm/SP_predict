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

	df = df.groupby(['quarter', 'ticker']).mean()
	df.to_csv("data/quarterly.csv")

def load_quarterly():
	df = pd.read_csv("data/quarterly.csv")
	df = df.reset_index(drop=True)
	df['quarter'] = pd.to_datetime(df['quarter'], infer_datetime_format=True)
	df = df.set_index('quarter')
	df['quarter'] =  df.index.to_period('Q')
	return df 

def load_changes():
	df = pd.read_csv("data/S_P_500_changes.csv")
	df['Remove_Date'] = pd.to_datetime(df['Remove Date'], infer_datetime_format=True)
	df = df.set_index('Remove_Date')
	df['Quarter_removed'] =  df.index.to_period('Q')
	df['Announcement Date'] = pd.to_datetime(df['Announcement Date'], infer_datetime_format=True)
	df = df.set_index('Announcement Date')
	df['Quarter_Annouced'] =  df.index.to_period('Q')
	df['Add Date'] = pd.to_datetime(df['Add Date'], infer_datetime_format=True)
	df = df.set_index('Add Date')
	df['Quarter_Added'] =  df.index.to_period('Q')
	df = df.reset_index()
	return df 

def join_dfs():
	quartely = load_quarterly()
	changes = load_changes()
	changes['ticker'] = changes['Stock Added']
	print df.head()
	return quartely.join(changes, on='ticker',rsuffix='changes')

def generate_sp_membership_list():
	membership_list = pd.read_csv('data/S&P_comp_20151209', header=None).values.flatten().tolist()
	df = load_changes()
	df = df.set_index('Quarter_Added')
	# create a list of quarter and all stocks in sp&500 set to fourth quarter 2015 
	quarter_membership_lists = [membership_list[:]  for i,row in enumerate(df.index.unique())] 
	quarter_order = df.index.unique().values.flatten().tolist()
	for i, q in enumerate(df.index.unique()):
		for k, row in enumerate(df.values):
			#we want to add and remove tickers from membership in the s&p 500 if the period is greater then or equal to the current period
			for x in xrange(i , quarter_order.index(row[-1].ordinal)):
				try:
					quarter_membership_lists[x].remove(row[2])
					break
				except ValueError:
					pass

				try:
					if row[1] not in quarter_membership_lists[x]:
						quarter_membership_lists[x].append(row[1])
					break
				except ValueError:
					pass
			pass	
	return quarter_order, quarter_membership_lists

def create_SP_500_member_df():
	quarter_order, quarter_membership_lists = generate_sp_membership_list()
	df = load_quarterly()
	SP_500_member = np.zeros((df.shape[0],1))
	print df['quarter']
	for i, row in df.iterrows():
		if row['quarter'].ordinal in quarter_order:
			if row['ticker'] in quarter_membership_lists[quarter_order.index(row['quarter'].ordinal)]:
				SP_500_member[i.to_period('Q').ordinal] = 1 
	df['SP_500_member'] = SP_500_member
	return df 

if __name__ == '__main__':
	# df = load_database()
	# df = process_database(df)
	# save_pivot(df)
	# generate_quarterly(df)
	# df = load_changes()
	# membership_list = generate_sp_membership_list()
	# quarter_order, quarter_membership_lists = generate_sp_membership_list()
	df = create_SP_500_member_df()
	#  print df.head()

