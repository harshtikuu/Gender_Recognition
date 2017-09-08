import pandas as pd
import numpy as np

def load_data():
	df=pd.read_csv('voice.csv')
	data=np.array(df.iloc[:])
	d={}
	d['data']=data[:,:-1]
	d['target']=data[:,-1]
	d['target'][d['target']=='male']=0
	d['target'][d['target']=='female']=1
	return d
