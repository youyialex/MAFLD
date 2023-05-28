import os,math
import pandas as pd 
import numpy as np
from icd_data_process import process_relative_data
    
def get_MAFLD(path):
    if os.path.exists('data_target/MAFLD1.csv'):
        pass
        
    else:
        if os.path.exists('data_target/FLI_MAFLD.csv'):
            
            FLI_data=pd.read_csv('data_target/FLI_MAFLD.csv',index_col='Participant ID')
        else:
            
            
            def FLI(triglycerides,ggt,waist_circumference,BMI):
                triglycerides=88.6*triglycerides 
                e_index=math.exp(0.935*math.log(triglycerides)+0.139*BMI+0.718*math.log(ggt)+0.053*waist_circumference-15.745)
                return 100*e_index/(1+e_index)
            
            FLI_data=pd.read_csv(path+'/MAFLD+糖尿病基线.csv',index_col='Participant ID')
            
            FLI_data['FLI再次']=FLI_data.apply(lambda x: FLI(triglycerides=x['甘油三酯'],
                                        ggt=x['GGT'],waist_circumference=x['腰围'],BMI=x['BMI'],
                                        ),axis=1)
            
            FLI_data.loc[(FLI_data['FLI再次']>60) & (FLI_data['代谢异常']==1),['MAFLD']]=1
            
            FLI_data=FLI_data.loc[:,['FLI再次','MAFLD']]
            FLI_data['MAFLD']=FLI_data['MAFLD'].fillna(0).astype(int)
            FLI_data.to_csv('data_target/FLI_MAFLD.csv')
        
        print('FLI_MAFLD数据维度\t:'+str(FLI_data.shape))
        print('MAFLD数量:\t'+str(sum(FLI_data['MAFLD']==1)))

        
        baseline_data=pd.read_csv(path+'/基线资料.csv',index_col=0)
        print('基线数据维度\t:'+str(baseline_data.shape))

        
        print('Townsend empty\t:'+str(
            sum(baseline_data['Townsend deprivation index at recruitment'].isna())
            ))    
        
        print('FLI empty\t:'+str(
        sum(FLI_data['FLI再次'].isna())
            ))

        
        baseline_data=baseline_data.join(FLI_data)
        
        baseline_data=baseline_data.loc[
            (baseline_data['Townsend deprivation index at recruitment'].notna())
            &(baseline_data['FLI再次'].notna())
            ,
            ['Sex','Age','Townsend deprivation index at recruitment','MAFLD','Drink','BMI25',]]
        
        print(f'shape:{baseline_data.shape[0]}')
        print('w/ MAFLD:'+str(sum(baseline_data['MAFLD']==1)))
        baseline_data.to_csv('data_target/MAFLD.csv')



def get_prs(path):
    if os.path.exists('data_target/MAFLD1.csv'):
        pass
        
    else:
        if os.path.exists('data_target/prs.csv'):
            prs=pd.read_csv('data_target/prs.csv',index_col=0)
        else:
            prs=pd.read_csv('data_original/PRS2.csv',index_col=0)
            tertile_low=prs['PRS'].quantile(1/3)
            tertile_high=prs['PRS'].quantile(2/3)
            prs['PRS']=prs['PRS'].apply(lambda x:0 if x<tertile_low else (1 if x>tertile_high else np.nan ) )
            
            prs=prs[['PRS']]
            prs[['PRS']].to_csv('data_target/prs.csv')   


        print(f'prs shape:{prs.shape[0]}')
        print(f'prs null:')
        print(sum(prs['PRS'].isna()))

        prs=prs.rename(columns={'PRS': 'MAFLD'})


        
        baseline_data=pd.read_csv(path+'/基线资料.csv',index_col='Participant ID')
        print('基线数据维度\t:'+str(baseline_data.shape))

        
        print('Townsend empty\t:'+str(
            sum(baseline_data['Townsend deprivation index at recruitment'].isna())
            ))   

        
        baseline_data=baseline_data.join(prs)
        
        baseline_data=baseline_data.loc[
            (baseline_data['Townsend deprivation index at recruitment'].notna())
            &(baseline_data['MAFLD'].notna())
            ,
            ['Sex','Age','Townsend deprivation index at recruitment','MAFLD','Drink','BMI25',]]
        
        print(f'final shape:{baseline_data.shape[0]}')
        print('w/ MAFLD:'+str(sum(baseline_data['MAFLD']==1)))
        baseline_data.to_csv('data_target/MAFLD.csv')


def psm_result():
    data=pd.read_csv('r_result/PSM匹配.csv',index_col=0)
    print(data.shape)
    death_info=process_relative_data()
    data['death']=death_info['40000-0.0'].notna().astype(int)
    data['fillin']=death_info['fillin'].dt.days
    print(data.shape)
    data.to_csv('r_result/psm_tags.csv')
    data=data.drop(columns=['distance','subclass','weights'])
    data=data[data['MAFLD']==1]
    data.to_csv('data_target/MAFLD_death.csv')
    print(data.shape)
    return 




def NFS_tag():
    data=pd.read_excel('data_original/NFS 纤维化数据.xlsx',index_col=1)
    baseline_data=pd.read_csv('r_result/psm_tags.csv',index_col=0)
    

    baseline_data=baseline_data.join(data[['NFS']])
    
    baseline_data['NFS_above']=(baseline_data['NFS']>=-1.455).astype(float)
    
    baseline_data['NFS_above']=baseline_data['NFS_above'].where(baseline_data['NFS_above']==1,np.nan)
    
    baseline_data['NFS_above']=baseline_data['NFS_above'].mask((baseline_data['NFS']<-1.455)&(baseline_data['MAFLD']==0),0)
    
    baseline_data['NFS_above']=baseline_data['NFS_above'].mask(baseline_data['NFS'].isna(),np.nan)


    baseline_data['NFS_below']=(baseline_data['MAFLD']&(baseline_data['NFS']<-1.455)).astype(float)
    baseline_data['NFS_below']=baseline_data['NFS_below'].where(baseline_data['NFS_below']==1,np.nan)

    baseline_data['NFS_below']=baseline_data['NFS_below'].mask((baseline_data['NFS']<-1.455)&(baseline_data['MAFLD']==0),0)
    baseline_data['NFS_below']=baseline_data['NFS_below'].mask(baseline_data['NFS'].isna(),np.nan)
    
    baseline_data.to_csv('r_result/psm_tags.csv')