import os
import pandas as pd
import numpy as np


def select_cols(df, cols):
    selected = []
    for x in df.columns.tolist():
        for y1 in cols:
            if y1 in x:
                selected.append(x)
    return selected

def predict_illness(df, disease_type= 'disease'):
    if disease_type == 'disease':
        name, time = '41270', '41280'
    elif disease_type == 'death':
        name, time = '40002', '40000'

    names = []
    times = []
    for x in df.columns.tolist():
        if name in x:
            names.append(x)
        if time in x:
            times.append(x)
    D_name = pd.DataFrame()
    D_time = pd.DataFrame()
    D_name= df[names].copy()
    D_time = df[times].copy()

    if disease_type == 'disease':
        D_time.columns = D_name.columns
        
        D_time[D_name.isna()]=np.nan 
        D_name[D_time.isna()]=np.nan 

    return D_name,D_time


def process_icd_data():
    if(os.path.exists('data_target/icd.pkl')):
        icd_data=pd.read_pickle('data_target/icd.pkl')
    else:        
        icd_data=pd.read_csv('data_original/ICD10_and_Death.csv',dtype=object,index_col=0)
        
        fields=['53-0','40000-0','40001-0','40002-0','41270','41280']
        icd_data=icd_data[select_cols(icd_data,fields)]
        
        icd_data=icd_data.rename(columns={'40001-0.0': '40002-0.0'})

        
        fields=['53-0','40000','41280']
        icd_data[select_cols(icd_data,fields)]=\
            icd_data[select_cols(icd_data,fields)].\
                apply(pd.to_datetime,format='%Y-%m-%d')
        
        icd_data.to_pickle('data_target/icd.pkl')  

    return icd_data


def process_relative_data():
    if os.path.exists('data_target/icd_data_relative.pkl'):
        icd_data=pd.read_pickle('data_target/icd_data_relative.pkl')
    else:
        icd_data=pd.read_pickle('data_target/icd.pkl')
        
        fields=['40002-0','41270']
        icd_data[select_cols(icd_data,fields)]=\
            icd_data[select_cols(icd_data,fields)].applymap(lambda x: x[0:3] if pd.notna(x) else np.nan)
        
        
        D=pd.read_excel("data_original/D1.xlsx",index_col=0)
        old_new_dict=D.set_index(['OLD'])['NEW'].to_dict()
        icd_data[select_cols(icd_data,['41270'])]=\
            icd_data[select_cols(icd_data,['41270'])].applymap(lambda x: old_new_dict[x] if x in old_new_dict else np.nan)
        
        old_death_dict=D.set_index(['OLD'])['DEATH_CODE'].to_dict()
        icd_data[select_cols(icd_data,['40002-0'])]=\
            icd_data[select_cols(icd_data,['40002-0'])].applymap(lambda x: old_death_dict[x] if x in old_death_dict else np.nan)
        
        
        icd_data['fillin']=np.datetime64('2022-02-02')
        
        icd_data['fillin']=icd_data[['fillin','40000-0.0']].min(axis=1)

        
        fields=['40000','41280','fillin']
        icd_data[select_cols(icd_data,fields)]=\
            icd_data[select_cols(icd_data,fields)].sub(icd_data['53-0.0'],axis=0)
        
        
        icd_data.to_pickle('data_target/icd_data_relative.pkl')
    return icd_data



def history_list(diseases):
    icd_data=pd.read_pickle('data_target/icd.pkl')
    fields=['40002-0','41270']
    icd_data[select_cols(icd_data,fields)]=\
    icd_data[select_cols(icd_data,fields)].applymap(lambda x: x[0:3] if pd.notna(x) else np.nan)

    disease,disease_time=\
    predict_illness(icd_data,disease_type='disease')
    result=pd.DataFrame()
    for D in diseases:
        days=(disease==D).any(axis=1).astype(int)
        result[D]=days

    result.to_csv('病史.csv')
    