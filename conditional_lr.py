import pandas as pd
import numpy as np
import math,os
from tqdm.auto import tqdm
from naming_scheme import type_and_suffix

def interval_process(df):
    df['interval']=df['OR'].apply(lambda x: "{:.2f}".format(x))+'('+\
    df['lower95'].apply(lambda x: "{:.2f}".format(x))+'-'+\
    df['upper95'].apply(lambda x: "{:.2f}".format(x))+')'
    df['frequency_bin']=df['frequency_bin'].fillna(0).astype(int).astype(str)
    df['same_day_percentage']=df['same_day_percentage'].apply(lambda x :"{:.2%}".format(x))

def generate_table(df_main,df_0,df_1):
    interval_process(df_main)
    interval_process(df_0)
    interval_process(df_1)
    
    return df_main[['frequency_bin','interval','same_day_percentage']]\
    .join(df_0[['frequency_bin','interval','same_day_percentage']],rsuffix='_0')\
    .join(df_1[['frequency_bin','interval','same_day_percentage']],rsuffix='_1')


def insert(df,i,df_add):
    
    df1 = df.iloc[:i, :]
    df2 = df.iloc[i:, :]
    df = pd.concat([df1, df_add, df2])
    return df





def get_matching_pairs(treated_df, non_treated_df, scaler=True,n_neighbors=2):
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors

    treated_x = treated_df.values
    non_treated_x = non_treated_df.values

    if scaler == True:
        scaler = StandardScaler()

    if scaler:
        scaler.fit(treated_x)
        treated_x = scaler.transform(treated_x)
        non_treated_x = scaler.transform(non_treated_x)

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(non_treated_x)
    indices = nbrs.kneighbors(treated_x, return_distance=False)
    treated_df['subclass']=-1
    non_treated_df['subclass']=-1
    
    for i in range(treated_df.shape[0]):
        treated_df.iloc[i,-1]=i
        non_treated_df.iloc[indices[i],-1]=i
    result=pd.concat([treated_df,non_treated_df])
    return result[result['subclass']>-1].copy()



def cal_lr(obj:type_and_suffix,n_neighbors,conditional=False):
    from statsmodels.discrete.conditional_models import ConditionalLogit
    from  statsmodels.api import Logit
    import warnings
    warnings.simplefilter(action='ignore', category=UserWarning)
    warnings.simplefilter(action='ignore', category=RuntimeWarning)
    os.makedirs('con_lr',exist_ok=True)

    _,for_cox=obj.read_cox_intermediate()

    pairs=pd.read_csv('binomial/binomial'+obj.get_suffix()+'.csv',index_col=0)

    con_lr=pd.DataFrame(columns=['D1','D2','OR','lower95','upper95','p_value','frequency'])
    con_lr.index.name='disease_pair'
    for Disease_pair in tqdm(pairs.index):
        [D1,D2]=Disease_pair.split(sep='_')
        try:
            
            
            for_cox['label']=0                
            
            both=((for_cox[D1+'_ICD']== True) & (for_cox[D2+'_ICD']==True))

            if obj.Death_cause!=None:
                both=both&(for_cox[obj.Death_cause+'_ICD']==True)
            
            
            both=both&((for_cox[D1+'_in']== True) & (for_cox[D2+'_in']==True))

            
            treated=both & ((for_cox[D1+'_days']-for_cox[D2+'_days'])<0)
            
            d2_gt_d1 =both & (~treated)
            
            for_cox.loc[treated,'label']=1

            include=((for_cox[D1+'_in']== True) & (for_cox[D2+'_in']==True)&(~d2_gt_d1))
            
            subgroup=for_cox.loc[include,['label',D1+'_ICD',D2+'_ICD','Sex','Age','Townsend.deprivation.index.at.recruitment']]
            
            treated_df=subgroup.loc[subgroup['label']==1,['Sex','Age','Townsend.deprivation.index.at.recruitment']]
            non_treated_df=subgroup.loc[subgroup['label']==0,['Sex','Age','Townsend.deprivation.index.at.recruitment']]

            
            results=get_matching_pairs(treated_df, non_treated_df, scaler=True,n_neighbors=n_neighbors)
            results=results.join(subgroup[[D1+'_ICD',D2+'_ICD']])
    
            if conditional:
                logit_model = ConditionalLogit(results[D2+'_ICD'], results[[D1+'_ICD']],groups=results['subclass'])
            else:
                logit_model = Logit(results[D2+'_ICD'], results[[D1+'_ICD']])

            model = logit_model.fit(disp=0)
            interval=np.exp(model.conf_int(alpha=0.05).loc[D1+'_ICD'])
            OR=np.exp(model.params[D1+'_ICD'])
            p_value=model.pvalues[D1+'_ICD']

            con_lr.loc[Disease_pair]=[D1,D2,OR,interval[0],interval[1],p_value,sum(results[D2+'_ICD']==1)]
        except:
            con_lr.loc[Disease_pair]=[D1,D2,np.nan,np.nan,np.nan,np.nan,0]
    

    
    if obj.subgroup=='all':
        threshold = 0.05/con_lr['frequency']
        con_lr=con_lr.loc[(con_lr['OR'] >= 1) & (con_lr['p_value'] < threshold), :]


    
    D1,D2=obj.disease_descrpition_dict()
    pairs['DD1']=pairs.index.map(lambda x : D1[x.split('_')[0]])
    if obj.Death_cause!=None:
        pairs['DD2']=pairs.index.map(lambda x : D1[x.split('_')[1]])  
    else:
        pairs['DD2']=pairs.index.map(lambda x : D2[x.split('_')[1]])  

    conditional='_c' if conditional else ''
    
    con_lr=con_lr.join(pairs,rsuffix='_bin')
    con_lr=con_lr.drop(columns=['p_value','frequency','unorder_frequency'])
    con_lr.to_csv('con_lr/T2'+conditional+obj.get_suffix()+'.csv')   

def paint_edge(data_original):
    data=data_original.copy(deep=True)
    data=data[['D1', 'D2','OR', 'frequency_bin', 'DD1', 'DD2']]
    result=pd.DataFrame()

    while(len(data)):
        source=set(data['D1'])
        target=set(data['D2'])
        src_nodes=source-target
        

        layer=data.loc[data['D1'].isin(src_nodes)]

        data=data.drop(index=layer.index)

        tgt_nodes=set(layer['D2'])-set(data['D2'])
        temp=layer[layer['D2'].isin(tgt_nodes)]
        result=pd.concat([result,temp])
        result=pd.concat([result,layer[~(layer['D1'].isin(temp['D1']))]])
        
    return result


def cal_lr_death(obj:type_and_suffix,n_neighbors):
    from  statsmodels.api import Logit
    import warnings
    warnings.simplefilter(action='ignore', category=UserWarning)
    warnings.simplefilter(action='ignore', category=RuntimeWarning)

    os.makedirs('Draw',exist_ok=True)

    _,for_cox=obj.read_cox_intermediate()

    pairs=pd.read_csv('con_lr/T2'+obj.get_suffix()+'.csv',index_col=0)
    pairs=pairs[['D1','D2','OR','frequency_bin','DD1','DD2']]
    Diseases=pairs['D2'].unique()
    con_lr=pd.DataFrame(columns=['D1','D2','OR','frequency_bin'])
    con_lr.index.name='disease_pair'

    D2=obj.Death_cause
    for D1 in tqdm(Diseases):
        frequency_bin=0
        try:
            
            
            for_cox['label']=0
            
            both=((for_cox[D1+'_ICD']== True) & (for_cox[D2+'_ICD']==True))
            
            both=both&((for_cox[D1+'_in']== True) & (for_cox[D2+'_in']==True))

            
            treated=both & ((for_cox[D1+'_days']-for_cox[D2+'_days'])<0)
            frequency_bin=sum(treated)
            
            d2_gt_d1 =both & (~treated)
            
            for_cox.loc[treated,'label']=1

            include=((for_cox[D1+'_in']== True) & (for_cox[D2+'_in']==True)&(~d2_gt_d1))
            
            subgroup=for_cox.loc[include,['label',D1+'_ICD',D2+'_ICD','Sex','Age','Townsend.deprivation.index.at.recruitment']]
            
            treated_df=subgroup.loc[subgroup['label']==1,['Sex','Age','Townsend.deprivation.index.at.recruitment']]
            non_treated_df=subgroup.loc[subgroup['label']==0,['Sex','Age','Townsend.deprivation.index.at.recruitment']]

            
            results=get_matching_pairs(treated_df, non_treated_df, scaler=True,n_neighbors=n_neighbors)
            results=results.join(subgroup[[D1+'_ICD',D2+'_ICD']])
    
          
            logit_model = Logit(results[D2+'_ICD'], results[[D1+'_ICD']])

            model = logit_model.fit(disp=0)
            OR=np.exp(model.params[D1+'_ICD'])

            con_lr.loc[D1+'_'+D2]=[D1,D2,OR,frequency_bin,]
        except:
            con_lr.loc[D1+'_'+D2]=[D1,D2,np.nan,0]

    
    D1,D2=obj.disease_descrpition_dict()
    con_lr['DD1']=con_lr.index.map(lambda x : D1[x.split('_')[0]])
    con_lr['DD2']=con_lr.index.map(lambda x : D2[x.split('_')[1]])
    
    con_lr=pd.concat([pairs,con_lr],axis=0)
    con_lr.to_csv('Draw/Death_original'+obj.get_suffix()+'.csv')
    paint_edge(con_lr).to_csv('Draw/Death_simplified'+obj.get_suffix()+'.csv')
    






        

































def create_table_subgroup(obj:type_and_suffix):
    os.makedirs('con_lr_subgroup/',exist_ok=True)
    writer=pd.ExcelWriter('con_lr_subgroup/T2_'+obj.disease_type+'.xlsx',engine='xlsxwriter')
    D1,D2=obj.subgroup_dict()
    for subtask in obj.subtasks:

        label=obj.subtasks[subtask]
        dfs=[type_and_suffix(obj.disease_type,obj.subgroup),
             type_and_suffix(obj.disease_type,subtask,0),
             type_and_suffix(obj.disease_type,subtask,1)]


        df_main=pd.read_csv('con_lr/T2'+dfs[0].get_suffix()+'.csv',index_col=0)
        df_0=pd.read_csv('con_lr/T2'+dfs[1].get_suffix()+'.csv',index_col=0)
        df_1=pd.read_csv('con_lr/T2'+dfs[2].get_suffix()+'.csv',index_col=0)
        
        overlapping=(~((df_0['lower95'] > df_1['upper95']) | (df_1['lower95'] > df_0['upper95']))).astype(int)
        result=generate_table(df_main,df_0,df_1)

        title=    ['All (N = '+dfs[0].get_cohort()[0]+')',
                   label[0]+' (N = '+dfs[1].get_cohort()[0]+')',
                   label[1]+' (N = '+dfs[2].get_cohort()[0]+')',]

        result.columns=pd.MultiIndex.from_product([title,['No.','HR (95% CI)','Percentage']])

        result['temp0']=result.index.str.split('_').str[0]
        result['temp1']=result.index.str.split('_').str[1]
        result.insert(0,'D1_D2 Code',result['temp0']+'_'+result['temp1'])
        result=result.sort_values(by=['temp0', 'temp1'], ascending=[True, True])
        result['Medical conditions']=result['temp0'].apply(lambda x: D1[x])+'*'+result['temp1'].apply(lambda x :D1[x])
        
        result['overlapping']=overlapping
        result=result.set_index('Medical conditions')
        
        
        
        
        
        
        
        
        
        
        
        
        result.index.name='Disease Pairs'
        result=result.drop(columns=[('temp0',''),('temp1','')])
        result.to_excel(writer, sheet_name=subtask)
        
        worksheet = writer.sheets[subtask]  
        max_len =result.index.astype(str).map(len).max()
        worksheet.set_column(0, 0, max_len) 
        for idx, col in enumerate(result):  
            series = result[col]
            max_len =series.astype(str).map(len).max()
            worksheet.set_column(idx+1, idx+1, max(max_len+1,5)) 



    writer.close()



