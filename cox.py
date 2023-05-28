import numpy as np
from datetime import timedelta
import pandas as pd 
import os 
from tqdm.auto import tqdm
from naming_scheme import type_and_suffix
import  icd_data_process


def interval_process(df,total):
    df['interval']=df['HR'].apply(lambda x: "{:.2f}".format(x))+'('+\
    df['lower95'].apply(lambda x: "{:.2f}".format(x))+'-'+\
    df['upper95'].apply(lambda x: "{:.2f}".format(x))+')'
    if total==True:
        df['frequency']=df['frequency'].fillna(0).astype(int).astype(str)+'/'+\
        df['total'].fillna(0).astype(int).astype(str)
    else:df['frequency']=df['frequency'].fillna(0).astype(int).astype(str)

def generate_table(df_main,df_0,df_1,total=False):
    interval_process(df_main,total)
    interval_process(df_0,total)
    interval_process(df_1,total)
    
    return df_main[['frequency','interval']]\
    .join(df_0[['frequency','interval']],rsuffix='_0')\
    .join(df_1[['frequency','interval']],rsuffix='_1')


def insert(df,i,df_add):
    
    df1 = df.iloc[:i, :]
    df2 = df.iloc[i:, :]
    df = pd.concat([df1, df_add, df2])
    return df




def get_disease_days(days, disease_name, base,frequency_count,fill_value,cohort,frequency):
    
    patients=days.dt.days>0
    fill=days.isna()
    
    if((patients&cohort).sum() > frequency): 
        frequency_count.loc[disease_name,'frequency']=(patients&cohort).sum()
        
        base[disease_name+'_in']=(patients|fill).astype(int)
        
        base[disease_name+'_ICD']=patients.astype(int)
        
        days[fill]=fill_value[fill]
        
        base[disease_name+'_days']=days.dt.days



def for_cox(obj:type_and_suffix,frequency_disease=None,frequency_death=None,):
    import warnings
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    os.makedirs('cox',exist_ok=True)

    
    if obj.subgroup!='all':
        return 
    if os.path.exists('data_target/for_cox.pkl'):
        return 
    
    
    group_size,(cohort,target_disease)=obj.get_cohort(range='all')
    icd_data=icd_data_process.process_relative_data()
    
    icd_data=icd_data.loc[cohort]
    
    fill_value=icd_data['fillin']

    
    D_death,D_disease=obj.disease_dict()
    
    for_cox=pd.DataFrame()
    

    
    if frequency_death==None:
        
        frequency_death=int(sum(target_disease==1)*0.005)
    print(f'freqency death:{frequency_death}')
    
    disease,disease_time=\
    icd_data_process.predict_illness(icd_data,disease_type='death')
    frequency_count=pd.DataFrame(columns=['frequency'])
    frequency_count.index.name = 'ICD'
    disease_time=disease_time['40000-0.0']
    for temp_disease in tqdm(D_death):
        
        days=disease_time.where((disease==temp_disease).any(axis=1),pd.NaT)
        get_disease_days(
                days, temp_disease, for_cox, frequency_count, fill_value,cohort=target_disease, frequency=frequency_death)
    frequency_count.sort_index().to_csv('cox/label_death_all.csv')


    
    if frequency_disease==None:
        frequency_disease=int(sum(target_disease==1)*0.01)
    print(f'freqency disease:{frequency_disease}')
    
    disease,disease_time=\
    icd_data_process.predict_illness(icd_data,disease_type='disease')
    frequency_count=pd.DataFrame(columns=['frequency'])
    frequency_count.index.name = 'ICD'
    for temp_disease in tqdm(D_disease):
        
        days=disease_time[disease==temp_disease].min(axis=1)
        get_disease_days(
            days, temp_disease, for_cox, frequency_count, fill_value,cohort=target_disease, frequency=frequency_disease)
    frequency_count.sort_index().to_csv('cox/label_disease_all.csv')

    
    for_cox=for_cox.join(target_disease)
    for_cox.to_pickle('data_target/for_cox.pkl')

def for_cox_sens(obj:type_and_suffix,):
    import warnings
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    os.makedirs('cox',exist_ok=True)

    
    if os.path.exists('data_target/for_cox_sens.pkl'):
        return 
    
    
    group_size,(cohort,target_disease)=obj.get_cohort(range='all')
    icd_data=icd_data_process.process_relative_data()
    
    icd_data=icd_data.loc[cohort]
    
    fill_value=icd_data['fillin']

    
    D_disease=pd.read_csv('cox/cox_result'+obj.get_cox_suffix()+'.csv',index_col=0).index
    
    for_cox=pd.DataFrame()
    

    
    disease,disease_time=\
    icd_data_process.predict_illness(icd_data,disease_type='disease')

    
    disease_death_dict=obj.disease_death_dict()
    disease_chapters=disease.copy(deep=True)
    disease_chapters=disease_chapters.applymap(lambda x: disease_death_dict[x] if x in disease_death_dict else np.nan)

    frequency_count=pd.DataFrame(columns=['frequency'])
    frequency_count.index.name = 'ICD'
    for temp_disease in tqdm(D_disease):
        
        days=disease_time[disease==temp_disease].min(axis=1)
        
        days_exclude=disease_time[disease_chapters==disease_death_dict[temp_disease]].min(axis=1)
        
        
        days=days.mask(days_exclude.dt.days<=0, days_exclude)

        get_disease_days(
            days, temp_disease, for_cox, frequency_count, fill_value,cohort=target_disease, frequency=0)


    
    Death=pd.read_csv('cox/cox_result_death_all.csv',index_col=0).index
    death,death_time=\
    icd_data_process.predict_illness(icd_data,disease_type='death')
    death_time=death_time['40000-0.0']

    for D in Death:
        dead=death_time.where((death==D).any(axis=1),pd.NaT)
        days_exclude=disease_time[disease_chapters==D].min(axis=1)
        dead=dead.mask(days_exclude.dt.days<=0, days_exclude)
        get_disease_days(
            dead, D, for_cox, frequency_count, fill_value,cohort=target_disease, frequency=0)


    
    for_cox=for_cox.join(target_disease)
    for_cox.to_pickle('data_target/for_cox_sens.pkl')




def cox_regression(obj:type_and_suffix,begin=None,end=None, target_disease='MAFLD',sens=False,custom=''):
    from lifelines import CoxPHFitter

    if sens:sens='_sens'
    else: sens=''

    
    group_size,(cohort,disease_label)=obj.get_cohort(range='all',label=target_disease)    
    data = pd.read_pickle('data_target/for_cox'+sens+'.pkl')
    data=data.loc[cohort]
    print(f'Cohort size:{group_size}')

    
    Diseases=pd.read_csv('cox/label'+obj.get_cox_suffix()+'.csv',index_col=0)
    
    result=pd.DataFrame(columns=['frequency','total','HR','lower95','upper95','p_value'])
    result.index.name='ICD'

    
    cph = CoxPHFitter()
    
    for D in tqdm(Diseases.index):
        try:
            index=data[data[D+'_in']==True].index
            cox_data=data.loc[index,[D+'_days',D+'_ICD']]
            cox_data[target_disease]=disease_label.astype(bool)
            if begin!=None:
                cox_data.loc[:,D+'_ICD']=(cox_data[D+'_days']>begin*360)&(cox_data[D+'_ICD'])
            if end != None:
                cox_data.loc[:,D+'_ICD']=(cox_data[D+'_days']<= end*360)&(cox_data[D+'_ICD'])
            
            
            frequency=sum(cox_data[D+'_ICD']&cox_data[target_disease])
            total=sum(cox_data[target_disease])
            cox=cph.fit(cox_data, duration_col=D+'_days', event_col=D+'_ICD', formula=target_disease).summary
            
            HR=cox.loc[target_disease,'exp(coef)']
            lower95=cox.loc[target_disease,'exp(coef) lower 95%']
            upper95=cox.loc[target_disease,'exp(coef) upper 95%']
            p_value=cox.loc[target_disease,'p']
            
            result.loc[D]=[frequency,total,HR,lower95,upper95,p_value]
        except: continue
    
    
    _,D1=obj.disease_descrpition_dict()
    
    Diseases=Diseases.drop(columns=['frequency'])
    Diseases=Diseases.join(result)
    Diseases['DESCRIPTION']=Diseases.index.map(lambda x : D1[x])

    
    if (begin==None)&(end==None)&(sens=='')&(custom==''):
        if (obj.subgroup=='all'):
            threshold=0.05/Diseases['frequency']
            
            Diseases=Diseases.loc[(Diseases['HR'] >= 1) & (Diseases['p_value'] < threshold), :]
        
    Diseases.to_csv('cox/cox_result'+obj.get_suffix()+sens+custom+'.csv')



def death_cox_regression(obj:type_and_suffix,custome=''):
    from lifelines import CoxPHFitter
    import warnings
    warnings.simplefilter(action='ignore', category=RuntimeWarning)

    Death_ICD=obj.Death_cause

    
    data = pd.read_pickle('data_target/for_cox.pkl')

    group_size,(cohort,MAFLD)=obj.get_cohort(range='all')
    print(f'Cohort size:{group_size}')
    
    data=data.loc[cohort]

    
    Diseases=pd.read_csv('cox/cox_result_disease_all.csv',index_col=0)
    
    result=pd.DataFrame(columns=['frequency','HR','lower95','upper95','p_value'])
    result.index.name='ICD'

    
    cph = CoxPHFitter()
    
    for D in tqdm(Diseases.index):

        try:
            index=data[data[D+'_in']==True].index
            cox_data=data.loc[index,[Death_ICD+'_days',Death_ICD+'_ICD',D+'_ICD']]

            
            frequency=sum(cox_data[D+'_ICD']&cox_data[Death_ICD+'_ICD']&data['MAFLD'])

            cox=cph.fit(cox_data, duration_col=Death_ICD+'_days', event_col=Death_ICD+'_ICD', formula=D+'_ICD').summary
            
            HR=cox.loc[D+'_ICD','exp(coef)']
            lower95=cox.loc[D+'_ICD','exp(coef) lower 95%']
            upper95=cox.loc[D+'_ICD','exp(coef) upper 95%']
            p_value=cox.loc[D+'_ICD','p']
            
            result.loc[D]=[frequency,HR,lower95,upper95,p_value]
        except: result.loc[D]=[0,np.nan,np.nan,np.nan,np.nan]
    
    
    D1,_=obj.disease_descrpition_dict()
    
    Diseases=Diseases[['frequency']].drop(columns=['frequency'])
    Diseases=Diseases.join(result)
    Diseases['DESCRIPTION']=Diseases.index.map(lambda x : D1[x])

    
    if obj.subgroup=='all':
        threshold=0.05/Diseases['frequency']
        
        Diseases=Diseases.loc[(Diseases['HR'] >= 1) & (Diseases['p_value'] < threshold), :]
        
    Diseases.to_csv('cox/cox_result'+obj.get_suffix()+'.csv')



def create_table_subgroup(obj:type_and_suffix):
    os.makedirs('cox_subgroup/',exist_ok=True)
    writer=pd.ExcelWriter('cox_subgroup/'+obj.disease_type+'.xlsx',engine='xlsxwriter')
    D1,D2=obj.subgroup_dict()
    for subtask in obj.subtasks:

        label=obj.subtasks[subtask]
        dfs=[type_and_suffix(obj.disease_type,obj.subgroup),
             type_and_suffix(obj.disease_type,subtask,0),
             type_and_suffix(obj.disease_type,subtask,1)]


        df_main=pd.read_csv('cox/cox_result'+dfs[0].get_cox_suffix()+'.csv',index_col=0)
        df_0=pd.read_csv('cox/cox_result'+dfs[1].get_suffix()+'.csv',index_col=0)
        df_1=pd.read_csv('cox/cox_result'+dfs[2].get_suffix()+'.csv',index_col=0)
        
        overlapping=(~((df_0['lower95'] > df_1['upper95']) | (df_1['lower95'] > df_0['upper95']))).astype(int)
        result=generate_table(df_main,df_0,df_1)

        title=    ['All (N = '+dfs[0].get_cohort()[0]+')',
                   label[0]+' (N = '+dfs[1].get_cohort()[0]+')',
                   label[1]+' (N = '+dfs[2].get_cohort()[0]+')',]

        result.columns=pd.MultiIndex.from_product([title,['No.','HR (95% CI)']])

        result.insert(0,'Code',result.index)
        result['Medical conditions']=result.index.map(lambda x : D1[x])
        
        result['overlapping']=overlapping
        result=result.set_index('Medical conditions')
        cols=result.columns
        
        if obj.disease_type=='disease':
            tag=''
            offset=0
            for i,(_,row) in enumerate(result.iterrows()):
                i+=offset
                if D2[row[('Code','')]]!=tag:
                    tag=D2[row[('Code','')]]
                    blank=pd.DataFrame([pd.Series([[''],[''],[''],[''],[''],[''],['']],)],columns=cols,index=['*'+tag+'*'])
                    result=insert(result,i,blank)
                    offset+=1
        result.index.name='Medical conditions'

        result.to_excel(writer, sheet_name=subtask)
        
        worksheet = writer.sheets[subtask]  
        max_len =result.index.astype(str).map(len).max()
        worksheet.set_column(0, 0, max_len) 
        for idx, col in enumerate(result):  
            series = result[col]
            max_len =series.astype(str).map(len).max()
            worksheet.set_column(idx+1, idx+1, max(max_len+1,5)) 



    writer.close()

        
def create_table_nfs(obj:type_and_suffix):
    os.makedirs('cox_subgroup/',exist_ok=True)
    writer=pd.ExcelWriter('cox_subgroup/nfs_'+obj.disease_type+'.xlsx',engine='xlsxwriter')
    D1,D2=obj.subgroup_dict()
    for subtask in obj.subtasks:


        df_main=pd.read_csv('cox/cox_result'+obj.get_suffix()+'.csv',index_col=0)
        df_0=pd.read_csv('cox/cox_result'+obj.get_suffix()+'_NFS_below.csv',index_col=0)
        df_1=pd.read_csv('cox/cox_result'+obj.get_suffix()+'_NFS_above.csv',index_col=0)
        
        overlapping=(~((df_0['lower95'] > df_1['upper95']) | (df_1['lower95'] > df_0['upper95']))).astype(int)
        result=generate_table(df_main,df_0,df_1)

        title=    ['All (N = '+obj.get_cohort()[0]+')',
                   'NFS_below'+' (N = '+obj.get_cohort(label='NFS_below')[0]+')',
                   'NFS_above'+' (N = '+obj.get_cohort(label='NFS_above')[0]+')',]

        result.columns=pd.MultiIndex.from_product([title,['No.','HR (95% CI)']])

        result.insert(0,'Code',result.index)
        result['Medical conditions']=result.index.map(lambda x : D1[x])
        
        result['overlapping']=overlapping

        result=result.set_index('Medical conditions')
        cols=result.columns
        
        if obj.disease_type=='disease':
            tag=''
            offset=0
            for i,(_,row) in enumerate(result.iterrows()):
                i+=offset
                if D2[row[('Code','')]]!=tag:
                    tag=D2[row[('Code','')]]
                    blank=pd.DataFrame([pd.Series([[''],[''],[''],[''],[''],[''],['']],)],columns=cols,index=['*'+tag+'*'])
                    result=insert(result,i,blank)
                    offset+=1
        result.index.name='Medical conditions'

        result.to_excel(writer, sheet_name=subtask)
        
        worksheet = writer.sheets[subtask]  
        max_len =result.index.astype(str).map(len).max()
        worksheet.set_column(0, 0, max_len) 
        for idx, col in enumerate(result):  
            series = result[col]
            max_len =series.astype(str).map(len).max()
            worksheet.set_column(idx+1, idx+1, max(max_len+1,5)) 



    writer.close()

        


def create_table_sens(obj:type_and_suffix,sens='_sens'):
    os.makedirs('cox_subgroup/',exist_ok=True)
    D1,D2=obj.subgroup_dict()

    df_main=pd.read_csv('cox/cox_result'+obj.get_suffix()+'.csv',index_col=0)
    df_0=pd.read_csv('cox/cox_result'+obj.get_suffix()+sens+'.csv',index_col=0)
    df_1=pd.read_csv('cox/cox_result'+obj.get_suffix()+sens+'.csv',index_col=0)

    df_0=df_0.loc[df_main.index]
    overlapping=(~((df_main['lower95'] > df_0['upper95']) | (df_0['lower95'] > df_main['upper95']))).astype(int)
    result=generate_table(df_main,df_0,df_1,total=True)

    title=    ['Main analysis',
                'Sensitivity analysis',
                'Sensitivity analysis(Duplicated)',]

    result.columns=pd.MultiIndex.from_product([title,['No./ N','HR (95% CI)']])
    
    result.insert(0,'Code',result.index)
    result['overlapping']=overlapping


    result['Medical conditions']=result.index.map(lambda x : D1[x])
    result=result.set_index('Medical conditions')
    cols=result.columns



    tag=''
    offset=0
    if obj.disease_type=='disease':
        for i,(_,row) in enumerate(result.iterrows()):
            i+=offset
            if D2[row[('Code','')]]!=tag:
                tag=D2[row[('Code','')]]
                blank=pd.DataFrame([pd.Series([[''],[''],[''],[''],[''],[''],['']],)],columns=cols,index=['*'+tag+'*'])
                result=insert(result,i,blank)
                offset+=1
    result.index.name='Medical conditions'
    result=result.drop(columns=[('Sensitivity analysis(Duplicated)','No./ N'),('Sensitivity analysis(Duplicated)','HR (95% CI)')])

    writer=pd.ExcelWriter('cox_subgroup/T10_'+obj.disease_type+sens+'.xlsx',engine='xlsxwriter')
    result.to_excel(writer, sheet_name='Sensitivity')

        
    worksheet = writer.sheets['Sensitivity']  
    max_len =result.index.astype(str).map(len).max()
    worksheet.set_column(0, 0, max_len) 
    for idx, col in enumerate(result):  
        series = result[col]
        max_len =series.astype(str).map(len).max()
        worksheet.set_column(idx+1, idx+1, max(max_len+1,5)) 

    writer.close()