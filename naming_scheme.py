import pandas as pd 
import icd_data_process
import copy

class type_and_suffix:
    def __init__(self,disease_type:str,subgroup:str,group_tag=None,custome=None,Death_cause=None) -> None:
        self.disease_type=disease_type
        self.subgroup=subgroup
        self.group_tag=group_tag
        self.custome=custome
        self.Death_cause=Death_cause
        self.subtasks={
                        'Sex':['Female','Male'],
                       'BMI25':['Fit','Fat'],
                       'Drink':['Not Drinking','Drinking'],

                       }
        
    def get_suffix(self):
        group_tag='_'+str(self.group_tag) if self.group_tag!=None else ''
        custome='_'+self.custome if self.custome!=None else ''
        if self.Death_cause!=None:
            return '_'.join(['',self.disease_type,self.Death_cause])
        else:
            return '_'.join(['',self.disease_type,self.subgroup,])+group_tag+custome

    def get_cox_suffix(self):
        if self.disease_type=='mix':
            return('_'.join(['','disease','all',]), 
                   '_'.join(['','death','all',]))
        else:
            return '_'.join(['',self.disease_type,'all',])


    def get_tag(self):
        data=pd.read_csv('r_result/psm_tags.csv', index_col=0,)
        if self.subgroup=='all':
            return data.index
        else:
            return  data[data[self.subgroup]==0].index, \
                    data[data[self.subgroup]==1].index


    def disease_dict(self):
        D=pd.read_excel("data_original/D1.xlsx")
        D1 = list(set(D['DEATH_CODE']))
        D2 = list(set(D['NEW']))      
        D1.sort()
        D2.sort()
        return D1,D2

    def disease_descrpition_dict(self):
        # 添加疾病的说明列
        D=pd.read_excel("data_original/D1.xlsx",index_col=0)

        D1=D.drop_duplicates(subset=['NEW'])
        D1=D1.set_index(['NEW'])['NEW_DESCRIPTION'].to_dict()
        if self.disease_type=='death':
            D2=D.drop_duplicates(subset=['DEATH_CODE'])
            D2=D2.set_index(['DEATH_CODE'])['DEATH_DESCRIPTION'].to_dict()
        else:
            D2=D1
        return D1,D2
    


    def disease_death_dict(self):
        # 添加疾病的说明列
        D=pd.read_excel("data_original/D1.xlsx",index_col=0)

        D1=D.drop_duplicates(subset=['NEW'])
        D1=D1.set_index(['NEW'])['DEATH_CODE'].to_dict()
        return D1

    #制表用字典
    def subgroup_dict(self):
        D=pd.read_excel("data_original/D1.xlsx",index_col=0)
        if self.disease_type=='disease':
            D=D.drop_duplicates(subset=['NEW'])
            D1=D.set_index(['NEW'])['NEW_DESCRIPTION'].to_dict()
            D2=D.set_index(['NEW'])['DEATH_DESCRIPTION'].apply(lambda x: x[:-6]).to_dict()
            return D1,D2
        else:
            D=D.drop_duplicates(subset=['DEATH_CODE'])
            D1=D.set_index(['DEATH_CODE'])['DEATH_DESCRIPTION'].apply(lambda x: x[:-6]).to_dict()
            return D1,0

    def get_cohort(self,range=None,label='MAFLD'):
        if self.Death_cause!=None:
            tag_data=pd.read_csv('r_result/PSM_death.csv', index_col=0)
            label='death'
        else:
            tag_data=pd.read_csv('r_result/psm_tags.csv', index_col=0)
            tag_data=tag_data.dropna(subset=[label])

        # 取出subgroup的数据
        if self.subgroup!='all':
            cohort=tag_data[tag_data[self.subgroup]==self.group_tag].index
        # 取出全部的数据
        else: cohort=tag_data.index
        tag_data=tag_data.loc[cohort]
        
        #如果是正常cox回归 则范围为所有人
        if range=='all':
            group_size=tag_data.shape[0]
        # 其他情况 则范围是是有MAFLD/death的人
        else:
            group_size=tag_data.shape[0]
            cohort=tag_data[tag_data[label]==1].index
            tag_data=tag_data.loc[cohort]

        # group_size=tag_data.shape[0]
        return str(group_size),(cohort,tag_data[label])

    

    def read_cox_intermediate(self):
        # 获取人群范围
        group_size,(tag,disease_label)=self.get_cohort()

        for_cox=pd.read_pickle('data_target/for_cox.pkl')

        if self.Death_cause==None:
            cox_index=pd.read_csv('cox/cox_result_disease_all.csv',index_col=0).index.to_list()
        else:
            cox_index=pd.read_csv('cox/cox_result'+self.get_suffix()+'.csv',index_col=0).index.to_list()

        if self.disease_type=='death':
            deaths=pd.read_csv('cox/cox_result_death_all.csv',index_col=0).index.to_list()
        else: deaths = ''
            
        for_cox=for_cox.loc[tag]

        # 添加用于条件逻辑回归时匹配的变量
        baseline_data=pd.read_csv('r_result/psm_tags.csv', index_col=0,)
        for_cox=for_cox.join(baseline_data[['Sex','Age','Townsend.deprivation.index.at.recruitment']])
        
        print(f'Cohort size:{for_cox.shape[0]}')


        if self.disease_type!='disease':
            frequency=icd_data_process.process_relative_data()
            frequency=frequency.loc[tag]
            frequency=int(sum(frequency['40000-0.0'].notna())*0.005)
        else:
            frequency=int(len(for_cox)*0.005)

        return (deaths,cox_index,frequency),for_cox.copy()
        
