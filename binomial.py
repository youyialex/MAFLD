import pandas as pd
import numpy as np
import math,os
from tqdm.auto import tqdm
from naming_scheme import type_and_suffix

def disease_pairs(obj:type_and_suffix,frequency=None,binomial=True):
    from scipy.stats import binomtest
    import warnings
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    os.makedirs('binomial',exist_ok=True)
    
    
    (deaths,cox_index,frequency_default),for_cox=obj.read_cox_intermediate()

    
    if frequency==None:
        frequency=frequency_default
    print(f'Frequency:{frequency}')

    for_lr=pd.DataFrame()

    base = pd.DataFrame(columns=['frequency','unorder_frequency','same_day_percentage'])
    base.index.name='disease_pair'
    p_unsatisfied=0
    f_unsatisfied=0
    for D1 in tqdm(cox_index):
        
        if obj.disease_type=='death': D2s=deaths
        else:  D2s=cox_index
        for D2 in D2s:
            
            indexes=((for_cox[D1+'_ICD']== True) & (for_cox[D2+'_ICD']==True))
            
            indexes=indexes&((for_cox[D1+'_in']== True) & (for_cox[D2+'_in']==True))
            
            total=sum((for_cox.loc[indexes,D1+'_days']-for_cox.loc[indexes,D2+'_days'])!=0)
            
            patients=(indexes & ((for_cox[D1+'_days']-for_cox[D2+'_days'])<0))
            
            D1_first=sum(patients)
            
            if D1_first>frequency:

                
                if binomial==True:
                    
                    threshold=0.05/D1_first
                    
                    same_day=1-(total/sum(indexes))
                    
                    bino=binomtest(D1_first,total,p=0.5, alternative='greater')
                    
                    if  bino.pvalue<threshold:
                        base.loc[D1+'_'+D2]=[D1_first,total,same_day]
                        for_lr[D1+'_'+D2]=patients.astype(int)
                    else:
                        p_unsatisfied+=1
                
                else :
                    same_day=1-(total/sum(indexes))
                    base.loc[D1+'_'+D2]=[D1_first,total,same_day]
                    for_lr[D1+'_'+D2]=patients.astype(int)

            else: f_unsatisfied+=1

    
    print(f'satisfied\tpairs:{base.shape[0]}\n\
p_unsatisfied\tpairs:{p_unsatisfied}\n\
f_unsatisfied\tpairs:{f_unsatisfied}\n\
total        \tpairs:{len(cox_index)*(len(D2s))}')
          
    base.to_csv('binomial/binomial'+obj.get_suffix()+'.csv')
    for_lr.to_csv('binomial/for_lr'+obj.get_suffix()+'.csv')



def disease_pairs_subgroup(obj:type_and_suffix):
    from scipy.stats import binomtest
    import warnings
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    
    
    _,for_cox=obj.read_cox_intermediate()

    pairs=pd.read_csv('con_lr/T2'+obj.get_cox_suffix()+'.csv',index_col=0)


    base = pd.DataFrame(columns=['frequency','unorder_frequency','same_day_percentage'])
    base.index.name='disease_pair'
    for Disease_pair in tqdm(pairs.index):
        try:
            [D1,D2]=Disease_pair.split(sep='_')
            
            indexes=((for_cox[D1+'_ICD']== True) & (for_cox[D2+'_ICD']==True))
            
            indexes=indexes&((for_cox[D1+'_in']== True) & (for_cox[D2+'_in']==True))
            
            total=sum((for_cox.loc[indexes,D1+'_days']-for_cox.loc[indexes,D2+'_days'])!=0)
            
            patients=(indexes & ((for_cox[D1+'_days']-for_cox[D2+'_days'])<0))
            
            D1_first=sum(patients)

            same_day=1-(total/sum(indexes))
            
            
            base.loc[D1+'_'+D2]=[D1_first,total,same_day]
        except:
            base.loc[D1+'_'+D2]=[0,0,0]

    
          
    base.to_csv('binomial/binomial'+obj.get_suffix()+'.csv')


def disease_death_pairs(obj:type_and_suffix,frequency=None):
    from scipy.stats import binomtest
    import warnings
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    
    
    (_,cox_index,frequency_default),for_cox=obj.read_cox_intermediate()

    if frequency==None:
        frequency=frequency_default
    
    for_lr = pd.DataFrame()
    base = pd.DataFrame(columns=['frequency','unorder_frequency','same_day_percentage'])
    base.index.name='disease_pair'

    p_unsatisfied=0
    f_unsatisfied=0
    for D1 in tqdm(cox_index):
        for D2 in cox_index:
            
            indexes=((for_cox[D1+'_ICD']== True) & (for_cox[D2+'_ICD']==True)& (for_cox[obj.Death_cause+'_ICD']==True))
            
            indexes=indexes&((for_cox[D1+'_in']== True) & (for_cox[D2+'_in']==True))
            
            total=sum((for_cox.loc[indexes,D1+'_days']-for_cox.loc[indexes,D2+'_days'])!=0)
            
            patients=(indexes &\
                       ((for_cox[D1+'_days']-for_cox[D2+'_days'])<0)& \
                        ((for_cox[D2+'_days']-for_cox[obj.Death_cause+'_days'])<0))
            
            D1_first=sum(patients)
            
            if D1_first>frequency:

                
                
                threshold=0.05/D1_first
                
                same_day=1-(total/sum(indexes))
                
                bino=binomtest(D1_first,total,p=0.5, alternative='greater')
                
                if  bino.pvalue<threshold:
                    base.loc[D1+'_'+D2]=[D1_first,total,same_day]
                    for_lr[D1+'_'+D2]=patients.astype(int)
                else:
                    p_unsatisfied+=1
                

            else: f_unsatisfied+=1

    
    print(f'satisfied\tpairs:{base.shape[0]}\n\
p_unsatisfied\tpairs:{p_unsatisfied}\n\
f_unsatisfied\tpairs:{f_unsatisfied}\n\
total        \tpairs:{len(cox_index)*(len(cox_index)-1)}')
        
    base.to_csv('binomial/binomial'+obj.get_suffix()+'.csv')
