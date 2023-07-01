#%%
import re
from tqdm import tqdm
import pandas as pd
from distfit import distfit
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from pandas import Series

#%%
patient=pd.read_csv("../data/patient.csv")

#%%
diagnose=pd.read_csv("../data/diagnosis.csv")
#%%
test=diagnose[0:100]
# 第一列：某个诊断的编号 第二列：用户入院id，第三列：该诊断出院时是否有效，第四列：确诊时间偏移，第五列:编码名称 6：icdcode 7 catalog

#%%
print(diagnose.isnull().sum())
# icd-9有15.9%为空
# 保留 2，6 7 ？ 5是否保留？
#%%
# 统计主诊断icdcode空值数量
test=diagnose
#%%
major=0
with tqdm(total=len(diagnose)) as pbar:
    for index,row in test.iterrows():
        if row['diagnosispriority']=='Major' or row['diagnosispriority']=='Primary':
            major=major+1
        pbar.update(1)

#%%
test=diagnose[0:100]
import numpy as np
major=0
major_nan=0
with tqdm(total=len(diagnose)) as pbar:
    for index,row in diagnose.iterrows():
        if (row['diagnosispriority']=='Major' or row['diagnosispriority']=='Primary') and (pd.isnull(row['icd9code'])):
            major_nan=major_nan+1
        pbar.update(1)

#%%
test=diagnose[['patientunitstayid','icd9code','diagnosispriority']]
#%%
#去掉空值
test=test.dropna(axis=0,subset = ["icd9code"])
#%%
#去掉重复值
test=test.drop_duplicates()
#%%
#保存一下
import pandas as pd
test=pd.read_csv("../data/tmp_diagnosis.csv",index_col=0)
#%%
# 对icd9编码处理，只保留，前的icd9编码
def prefix(a):
    temp_drug = a.strip()
    temp_drug = temp_drug[:temp_drug.find(",")]
    return temp_drug

row=test[:1]

#%%
for index,row in test.iterrows():
    test.at[index,'icd9code']=prefix(test.at[index,'icd9code'])

#%%
test=test.drop_duplicates()
#%%
test.to_csv("../data/tmp_diagnosis.csv",index=0)
#%%
# 读入医院数据，与诊断结果进行连接，对每个医院诊断结果进行分布统计
patient=pd.read_csv("../data/patient.csv")
test=pd.read_csv("../data/tmp_diagnosis.csv")
#%%
hospital_record=patient[['patientunitstayid','hospitalid']]
#%%
#病人医院信息与diagnosis连接
diagnosis=test.merge(hospital_record,how='inner',on='patientunitstayid')
#%%
diagnosis = diagnosis.sort_values(by=["hospitalid", "icd9code"], ascending=[True, True])
diagnosis=diagnosis.reset_index(drop=True)
#%%
counts=test['hospitalid'].value_counts()
#%%
#统计医院诊断数
num=0
for i in counts:
    if i>1000:
        num=num+1
#%%
#对整体数据分布先分析一下
drop_idex=[]
for index,row in diagnosis.iterrows():
    if re.findall('^[A-Z].*',row['icd9code'])!=[]:
        drop_idex.append(index)
#%%
diagnosis=diagnosis.drop(index=drop_idex)

X = diagnosis['icd9code']
dist = distfit(todf=True)
dist.fit_transform(X)
dist.plot()
plt.show()
#%%
diagnosis.to_csv("../data/tmp_diagnosis.csv",index=0)
#%%
hospital_list=[]
for index,row in diagnosis.iterrows():
    if row['hospitalid'] not in hospital_list:
        hospital_list.append(row['hospitalid'])
#%%
list=sorted(hospital_list)

#%%
for id in list:
    tmp=diagnosis[diagnosis['hospitalid']==id]
    X=tmp['icd9code']
    dist=distfit(todf=True)
    dist.fit_transform(X)
    dist.plot()
    f = plt.gcf()  # 获取当前图像
    f.savefig(r'../pre-process/diagnosis_dist/client_dist/hospital_{}.png'.format(id))
    f.clear()  # 释放内存
#%%
#----------------------开始对同一诊断药品开具情况进行分析---------------------------
diagnosis=pd.read_csv("../data/tmp_diagnosis.csv")
patient=pd.read_csv("../data/patient.csv")
#%%
test=diagnosis[['patientunitstayid','icd9code']]
#%%
# sort
test = test.sort_values(by=["patientunitstayid", "icd9code"], ascending=[True, True])

#%%
from pandas import Series

df_group = test.groupby(['patientunitstayid'])
max_count_name =df_group.count().sort_values(by='icd9code',ascending=False)['icd9code'].tolist()[0] #找到出现最多的有几个
new_df_colname = ['patientunitstayid']+[f'diagnose{number+1}' for number in range(max_count_name)]  #生成新的df的列名
process_diagnose = pd.DataFrame(columns=new_df_colname)
num=test['patientunitstayid'].nunique()


with tqdm(total=num) as pbar:
    for i in df_group:
        # print(i[0])
        # print(i[1]['icd9code'].tolist())
        data = [i[0], *i[1]['icd9code'].tolist()]
        ser = Series(data,new_df_colname[:len(data)])
        process_diagnose=process_diagnose.append(ser,ignore_index=True)
        # break
        pbar.update(1)


#%%
process_diagnose=process_diagnose.fillna(0)
#%%
process_diagnose.to_csv("../data/process_diagnose.csv")
#%%

df_group = test.groupby(['patientunitstayid'])
max_count_name =df_group.count().sort_values(by='icd9code',ascending=False)['icd9code'].tolist()[0] #找到出现最多的有几个
new_df_colname = ['patientunitstayid','diagnose']  #生成新的df的列名
process_diagnose = pd.DataFrame(columns=new_df_colname)
num=test['patientunitstayid'].nunique()


with tqdm(total=num) as pbar:
    for i in df_group:
        # print(i[0])
        # print(i[1]['icd9code'].tolist())
        data = [i[0], i[1]['icd9code'].tolist()]
        ser = Series(data,new_df_colname[:len(data)])
        process_diagnose=process_diagnose.append(ser,ignore_index=True)
        # break
        pbar.update(1)

#%%
process_diagnose.to_csv("../data/diagnosis_list.csv",index=0)
#%%
# 寻找相同诊断的病人
process_diagnose=pd.read_csv("../data/diagnosis_list.csv")
process_diagnose.drop('Unnamed: 0',axis=1,inplace=True)
#%%
key_duplicated=process_diagnose[process_diagnose.diagnose.duplicated(False)]
sort_diagnose_duplicated=key_duplicated.sort_values(by=['diagnose'])
#%%
df_group = sort_diagnose_duplicated.groupby(['diagnose'])
new_df_colname = ['diagnose','patientunitstayid']  #生成新的df的列名
dupliated_patient = pd.DataFrame(columns=new_df_colname)
num=df_group['diagnose'].nunique()

with tqdm(total=len(num)) as pbar:
    for i in df_group:
        # print(i)
        # print(i[0])
        # print(i[1]['patientunitstayid'].tolist())
        data = [i[0], i[1]['patientunitstayid'].tolist()]
        ser = Series(data,new_df_colname[:len(data)])
        dupliated_patient=dupliated_patient.append(ser,ignore_index=True)
        # break
        pbar.update(1)
#%%
dupliated_patient.to_csv("../data/dupliated_patient.csv",index=0)

#%%
#----------------------------读入药品表，开始对用药情况进行统计处理--------------------------------------
medication=pd.read_csv("../data/medication.csv")
#%%
test=medication[0:100]
#%%
medication=medication[['patientunitstayid','drugstartoffset','drugordercancelled','drugname','gtc','drugstopoffset']]
#%%
medication=medication[medication['drugordercancelled']=='No']
#%%
medication=medication[['patientunitstayid','drugname','drugstartoffset','drugstopoffset']]
#%%
medication=medication.drop_duplicates()
#%%
medication=medication.dropna(axis=0,subset = ["drugname"])   # 丢弃DRUG中有缺失值的行
#%%
drugname=medication['drugname'].value_counts()
drugname=drugname.to_frame()
drugname=drugname.reset_index()
drugname.columns = ['drugname','counts']
drugname=drugname.sort_values(by=['drugname'])

#%%
drugname.to_csv("../data/ddddddd.csv",index=0)
#%%
medication['drugname']=medication['drugname'].str.lower()
#%%
medication=medication.drop_duplicates()
#%%
drug_dict={
    '<<norepinephrine':'norepinephrine',
    'alum & mag hydroxide-simeth 200-200-20 mg/5ml po susp':'alum & mag hydroxide-simeth',
    'buffered lidocaine 1%':'buffered lidocaine',
    'fluticasone-salmeterol 250-50 mcg/dose in aepb':'fluticasone-salmeterol',
    'influenza vac split quad 0.5 ml im susp':'influenza vac split quad',
    'k phos mono-sod phos di & mono 155-852-130 mg po tabs':'k phos mono-sod phos di & mono',
    'lactated ringer\'s 1,000 ml bag':'lactated ringer\'s',
    'lactated ringers':'lactated ringer\'s',
    'pnu-immune-23':'pneumococcal 23-valps vaccine',
    'potassium & sodium phosphates 280-160-250 mg po pack':'potassium and sodium phosphate',
    'potassium cl in water 20meq/50':'potassium chloride',
    'sodium chl':'sodium chloride',
    'sodium cl':'sodium chloride',
}
#%%
drug_dict={
    'vitamins/minerals po tabs':'vitamins'
}

for idx, name in enumerate(drugname['drugname']):
    for new_name in drugname['drugname'][idx+1:]:
        if name in new_name:
            drug_dict[new_name]=name
        else:
            break
#%%
drug_dict={
    'kcl 10 meq/100 ml':'kcl'
}

for idx, name in enumerate(drugname['drugname']):
    for new_name in drugname['drugname'][idx+1:]:
        if name in new_name:
            drug_dict[new_name]=name
        else:
            break
#%%
drug_dict={
    'lactated ringers ection':'lactated ringer\'s',
    'lactated ringers iv soln':'lactated ringer\'s',
    'lactated ringers iv solp':'lactated ringer\'s'
}
#%%
for idx, name in enumerate(drugname['drugname']):
    for new_name in drugname['drugname'][idx+1:]:
        if name in new_name:
            drug_dict[new_name]=name
        else:
            break
#%%
# 在drug表操作
with tqdm(total=len(drugname)) as pbar:
    for index,row in drugname.iterrows():
        name=row['drugname']
        if name in drug_dict:
            drugname.at[index,'drugname']=drug_dict[name]
        pbar.update(1)
drugname=drugname.drop_duplicates()
#%%
# 在medication表操作
with tqdm(total=len(medication)) as pbar:
    for index,row in medication.iterrows():
        name=row['drugname']
        if name in drug_dict:
            medication.at[index,'drugname']=drug_dict[name]
        pbar.update(1)
medication=medication.drop_duplicates()

#%%
import re
# 对 unit+： 和unit+-进行处理
def preprocess(text):
    temp_drug = text.strip()
    temp_drug = re.sub(r'(.+?).*\s-\s', '', temp_drug)
    temp_drug = re.sub(r'(.+?).*:\s', '', temp_drug)
    temp_drug = " ".join(temp_drug.split())
    temp_drug = temp_drug.strip()

    return temp_drug

#%%
for index,row in drugname.iterrows():
    #print(drug.at[index,'DRUG'])
    drugname.at[index,'drugname']=preprocess(drugname.at[index,'drugname'])

drugname=drugname.drop_duplicates()
drug_name=drugname['drugname']
drug_name=drug_name.drop_duplicates()
#%%
with tqdm(total=len(medication)) as pbar:
    for index, row in medication.iterrows():
        # print(drug.at[index,'DRUG'])
        medication.at[index, 'drugname'] = preprocess(medication.at[index, 'drugname'])
        pbar.update(1)
medication=medication.drop_duplicates()

#%%
medication=pd.read_csv("../data/pro_medication.csv")
#%%
# again 正则 2 mg 4 % 250-50 mcg (per unit) *unit* inj 2g
import re

def preprocess(text):
    temp_drug = text.strip()
    temp_drug = re.sub(r' [0-9]+ \d*mg(.+?).*', '', temp_drug)
    temp_drug = re.sub(r' [0-9]+\d*mg(.+?).*', '', temp_drug)
    temp_drug = re.sub(r' [0-9]+.\d* mg(.+?).*', '', temp_drug)
    temp_drug = re.sub(r' [0-9]+.\d*mg(.+?).*', '', temp_drug)
    temp_drug = re.sub(r' [0-9]+ \d*mg', '', temp_drug)

    temp_drug = re.sub(r' [0-9]+ \d*units(.+?).*', '', temp_drug)

    temp_drug = re.sub(r' [0-9]+ \d*%(.+?).*', '', temp_drug)
    temp_drug = re.sub(r' [0-9]+.\d*%(.+?).*', '', temp_drug)
    temp_drug = re.sub(r' [0-9]+.\d* %(.+?).*', '', temp_drug)

    temp_drug = re.sub(r' [0-9]+ \d*%', '', temp_drug)
    temp_drug = re.sub(r'[0-9]+ \d*%', '', temp_drug)
    temp_drug = re.sub(r'[0-9]+.\d*%', '', temp_drug)

    temp_drug = re.sub(r' [0-9]+ \d*mcg(.+?).*', '', temp_drug)
    temp_drug = re.sub(r' [0-9]+\d*mcg(.+?).*', '', temp_drug)

    # temp_drug = re.sub(r' [0-9\]+\d-[0-9]+.\d* g(.+?).*', '', temp_drug)
    temp_drug = re.sub(r' [0-9]+ \d*g(.+?).*', '', temp_drug)

    temp_drug = re.sub(r'\((per unit\))', '', temp_drug)
    temp_drug = temp_drug.replace("*unit* inj", "")

    temp_drug = " ".join(temp_drug.split())
    temp_drug = temp_drug.strip()

    return temp_drug
#%%
for index,row in drugname.iterrows():
    #print(drug.at[index,'DRUG'])
    drugname.at[index,'drugname']=preprocess(drugname.at[index,'drugname'])

drugname=drugname.drop_duplicates()
drug_name=drugname['drugname']
drug_name=drug_name.drop_duplicates()
#%%
def preprocess(text):
    temp_drug = text.strip()
    temp_drug = temp_drug.replace("inj", "")
    temp_drug = temp_drug.replace("hcl", "")
    temp_drug = temp_drug.replace("(human)", "")
    temp_drug = temp_drug.replace("(pf)", "")

    temp_drug = " ".join(temp_drug.split())
    temp_drug = temp_drug.strip()

    return temp_drug

#%%
medication=medication.drop( index = medication.drugname[medication.drugname == "15 ml cup"].index )
#%%
drugname.to_csv("../data/drugname.csv",index=0)
#%%
# 给药品以编号
drugcode=pd.read_csv("../data/drugname.csv")
#%%
medication=medication.merge(drugcode,how='inner',on='drugname')
#%%
medication=medication[['patientunitstayid','drugstartoffset','drugstopoffset','counts']]
#%%
medication.columns = ['patientunitstayid','drugstartoffset','drugstopoffset','code']
#%%
# 将带有时间的medication表进行保存
medication.to_csv("../data/drug_time.csv",index=0)
#%%
medication = medication.sort_values(by=["patientunitstayid", "code"], ascending=[True, True])
#%%
df_group = medication.groupby(['patientunitstayid'])
max_count_name =df_group.count().sort_values(by='code',ascending=False)['code'].tolist()#找到出现最多的有几个
#%%
count_df=pd.DataFrame(max_count_name)
#%%
new_df_colname = ['patientunitstayid','drug']  #生成新的df的列名
process_drug = pd.DataFrame(columns=new_df_colname)
num=medication['patientunitstayid'].nunique()

with tqdm(total=num) as pbar:
    for i in df_group:
        # print(i[0])
        # print(i[1]['code'].tolist())
        data = [i[0], i[1]['code'].tolist()]
        ser = Series(data,new_df_colname[:len(data)])
        process_drug=process_drug.append(ser,ignore_index=True)
        # break
        pbar.update(1)
#%%
process_drug.to_csv("../data/drug_list.csv",index=0)
#%%
diagnosis_list=pd.read_csv("../data/diagnosis_list.csv")
#%%
prescription=diagnosis_list.merge(process_drug,how='inner',on='patientunitstayid')
#%%
prescription=prescription.sort_values(by=["diagnose"], ascending=[True])
#%%
with tqdm(total=len(prescription)) as pbar:
    for index,row in prescription.iterrows():
        prescription.at[index,'diagnose']=str(row['diagnose'])
        prescription.at[index,'drug']=str(row['drug'])
        pbar.update(1)



#%%
duplicate=prescription.duplicated(subset=['diagnose','drug'],keep=False)
#%%
duplicate=prescription[prescription.duplicated(subset=['diagnose','drug'],keep=False)]
#%%
duplicate=duplicate.sort_values(by=["diagnose",'drug'], ascending=[True,True])
#%%
patient=pd.read_csv("../data/patient.csv")
#%%
duplicate.to_csv("../data/duplicate.csv",index=0)
#%%
duplicate=duplicate.merge(patient,how='inner',on='patientunitstayid')
#%%
duplicate=duplicate[['patientunitstayid','diagnose','drug','hospitalid']]
#%%
duplicate=duplicate.sort_values(by=["diagnose",'drug','hospitalid'], ascending=[True,True,True])
#%%
prescription.to_csv("../data/prescription.csv",index=0)
#%%
#-----------------------------对化验结果处理---------------------------------------
lab=pd.read_csv("../data/lab.csv")
#%%
test=lab[0:100]
#%%
count=lab['labname'].value_counts()
#%%
count=count.to_frame()
count=count.reset_index()
count.columns = ['labname','counts']
count=count.sort_values(by=['labname'])
#%%
count.to_csv("../data/lab_name.csv",index=0)
#%%
lab=lab[['patientunitstayid', 'labresultoffset','labname','labresult']]
#%%
labname=pd.read_csv("../data/lab_name.csv")
#%%
prescription=pd.read_csv("../data/prescription.csv")
#%%
patientunitstayid=prescription['patientunitstayid']
#%%
patientunitstayid=pd.read_csv("../data/patientunitstayid.csv")
#%%
lab=lab.merge(patientunitstayid,how='inner',on='patientunitstayid')
#%%
lab=lab.merge(labname,how='inner',on='labname')
#%%
lab=lab[['patientunitstayid', 'labresultoffset','labid','labresult']]
#%%
new_df_colname = ['patientunitstayid', 'labresultoffset','labid','labresult']
lab.columns=new_df_colname
#%%
lab=lab.dropna(axis=0,subset = ["labresult"])
#%%
lab=lab.drop_duplicates()
#%%
lab.to_csv("../data/feature/lab.csv",index=0)
#%%
medication=pd.read_csv("../data/drug_time.csv")
lab=pd.read_csv("../data/lab.csv")
#%%
test_m=medication[:100]
test_l=lab[:100]
#%%
patient_l=lab['patientunitstayid'].drop_duplicates()
patient_m=test['patientunitstayid'].drop_duplicates()
#%%
patient_l=patient_l.to_frame()
patient_m=patient_m.to_frame()
#%%
id=patient_l.merge(patient_l,how='inner')
#%%
medication=medication.merge(id,how='inner',on='patientunitstayid')
#%%
lab=lab.sort_values(by=['patientunitstayid','labresultoffset'], ascending=[True, True])
#%%
medication = medication.sort_values(by=['patientunitstayid',"drugstartoffset", "drugstopoffset"], ascending=[True, True, True])
#%%
lab=lab.drop_duplicates()
medication=medication.drop_duplicates()
#%%
medication.to_csv("../data/drug_time.csv",index=0)
lab.to_csv("../data/lab.csv",index=0)
#%%
time_list={}
patient=lab['patientunitstasyid'].drop_duplicates()
#%%
df_group = lab.groupby(['patientunitstayid'])
#%%
with tqdm(total=len(df_group)) as pbar:
    for i in df_group:
        print(i[0])
        print(i[1]['labresultoffset'].drop_duplicates().tolist())

        break
        pbar.update(1)

#%%
patientunitstayid=pd.read_csv('../data/feture/patientunitstayid.csv')
#%%
patient=patient.merge(patientunitstayid,how='inner',on='patientunitstayid')

#%%
import numpy as np

test=patient[['hospitaldischargestatus','unitdischargestatus']]
test['result']=np.where(test['hospitaldischargestatus']==test['unitdischargestatus'],'1','0')

#%%
patient=patient[['patientunitstayid','gender','age','ethnicity','hospitalid','admissionheight','admissionweight','hospitaldischargestatus']]
#%%
patient['age']=patient['age'].apply(lambda x : 90 if x == '> 89' else x)
#%%
patient['age']=patient['age'].astype('int')
#%%
mid=patient['age'].median()
#%%
patient['age'].fillna(mid,inplace=True)
#%%
def age_class(x):
    y=0
    if x<18:
        y=0
    elif x>=18 and x<=45:
        y=1 #youth
    elif x>45 and x<=59:
        y=2 # middle age
    else:
        y=3 # agedness`
    return y

patient['age_class']=" "
patient['age_class']=patient.apply(lambda row: age_class(row['age']),axis=1)
#%%
labelencoder = LabelEncoder()
patient['gender'] = labelencoder.fit_transform(patient['gender'])
#%%
patient['ethnicity'] = labelencoder.fit_transform(patient['ethnicity'])
#%%
patient['hospitaldischargestatus'] = labelencoder.fit_transform(patient['hospitaldischargestatus'])
#%%
test=patient[['patientunitstayid', 'gender', 'age_class', 'ethnicity', 'hospitalid',
       'admissionheight', 'admissionweight', 'hospitaldischargestatus',]]
#%%
test.to_csv("../data/feature/patient.csv")
#%%
diagnosis=pd.read_csv('../data/process_diagnose.csv')
#%%
diagnosis=diagnosis.drop(['Unnamed: 0'],axis=1)
#%%
diagnosis=diagnosis.merge(patientunitstayid,how='inner',on='patientunitstayid')
#%%
diagnosis=diagnosis.drop(['flag'],axis=1)
#%%
diagnosis.to_csv("../data/feature/diagnosis.csv")
#%%
lab=pd.read_csv("../data/lab.csv")
diagnosis=pd.read_csv('../data/feature/diagnosis.csv')
patient=pd.read_csv('../data/feature/patient.csv')

#%%
test=lab.head()
#%%
labid=pd.read_csv('../data/lab_name.csv')
#%%
lab_dict={'1':'1',}
for index,row in labid.iterrows():
    lab_dict[row['id1']]=row['id2']
#%%
with tqdm(total=len(lab)) as pbar:
    for index,row in lab.iterrows():
        id=row['labid']
        lab.at[index,'labid']=lab_dict[id]
        pbar.update(1)
#%%
labid=labid[['labid','labid2']]
#%%
lab=lab.merge(labid,how='inner',on='labid')
#%%
lab=lab[['patientunitstayid', 'labresultoffset', 'labid2', 'labresult']]
#%%
test=lab.head()
#%%
lab.to_csv("../data/lab.csv",index=0)
#%%
lab2=pd.read_csv("../data/lab.csv")
#%%
test=lab[:1000]
#%%
lab_state=pd.read_csv("../data/feature/lab.csv")
#%%
pro_test=lab_state[:1000]
#%%
drop_list=[1,6,10,22,27,29,33,41,42,43,49,57,59,61,66,70,71,75,76,77,78,95,96,99,101,155]
#%%
drop_index=[]
with tqdm(total=len(lab)) as pbar:
    for index,row in lab.iterrows():
        tmp=row['labid']
        if tmp in drop_list:
            drop_index.append(index)
        pbar.update(1)
#%%
#-------删除无法度量的指标-------
lab=lab.drop(index=drop_index)
#%%
lab_patientid=drop_lab2['patientunitstayid'].drop_duplicates()
#%%
lab_patientid.to_csv("../data/feature/patientunitstayid.csv")
#%%
lab_refer=pd.read_csv('../data/lab_reference.csv')
#%%
labid=lab_refer[['labname','labid']]
#%%
patient=pd.read_csv('../data/feature/patient.csv')
#%%
patientunitstayid=pd.read_csv('../data/feature/patientunitstayid.csv')
#%%
patient=patient.merge(patientunitstayid,how='inner',on='patientunitstayid')
#%%
lab_refer=lab_refer[['labid','gender','lower','upper']]
#%%
lab_patient=patient[['patientunitstayid','gender']]
#%%
lab=lab.merge(lab_patient,how='inner',on='patientunitstayid')
#%%
drop_lab2['label']=''
#%%
lab2=lab.drop(['lower','upper'],axis=1)
#%%
lab2=lab2.merge(lab_refer,how='inner',on=['labid','gender'])
#%%
test=lab2[:1000]
#%%
drop_lab2=drop_lab2[['patientunitstayid', 'labresultoffset', 'labresult','labid', 'lower', 'upper', 'label']]
#%%
#------删除空值
drop_lab=drop_lab2.dropna(axis=0,subset = ["labresult"])
#%%
with tqdm(total=len(drop_lab)) as pbar:
    for index,row in drop_lab.iterrows():
        if ((row['labresult']>=row['lower'] and row['labresult']<=row['upper'])):
            drop_lab.at[index,'label']=0
        else:
            drop_lab.at[index,'label']=1
        pbar.update(1)
#%%
drop_lab.to_csv("../data/feature/lab_label2.csv",index=False)
#%%
lab=pd.read_csv("../data/feature/lab_label.csv")
#%%
lab.to_csv("../data/feature/lab.csv",index=False)
#%%
lab_label=lab[['patientunitstayid','labresultoffset','labid','label']]
#%%
labid=lab_label['labid'].drop_duplicates()
#%%
from pandas import Series

df_group = lab_label.groupby(['patientunitstayid','labresultoffset'])
max_count_name =df_group.count().sort_values(by='labid',ascending=False)['labid'].tolist()[0] #找到出现最多的有几个
new_df_colname = ['patientunitstayid']+[f'diagnose{number+1}' for number in range(max_count_name)]  #生成新的df的列名
process_diagnose = pd.DataFrame(columns=new_df_colname)
num=test['patientunitstayid'].nunique()
#%%

with tqdm(total=num) as pbar:
    for i in df_group:
        # print(i[0])
        # print(i[1]['icd9code'].tolist())
        data = [i[0], *i[1]['icd9code'].tolist()]
        ser = Series(data,new_df_colname[:len(data)])
        process_diagnose=process_diagnose.append(ser,ignore_index=True)
        # break
        pbar.update(1)

#%%
lab2=lab2[['patientunitstayid', 'labresultoffset','labname','labresult']]
#%%
labname=pd.read_csv('../data/lab_name.csv')
#%%
labname=labname[['labname','labid']]
#%%
lab2=lab2.merge(labname,how='inner',on='labname')
#%%
lab2=lab2.merge(patientunitstayid,how='inner',on='patientunitstayid')
#%%
lab2=lab2[['patientunitstayid', 'labresultoffset', 'labname', 'labresult']]
lab2=lab2.merge(lab_patient,how='inner',on='patientunitstayid')
#%%
lab2=lab2[['patientunitstayid', 'labresultoffset', 'labname', 'labresult','gender']]
#%%
lab_refer=lab_refer[['labname','labid','gender','lower','upper']]
#%%
drop_lab2=lab2.merge(lab_refer,how='inner',on=['labname','gender'])
#%%
#-----完成从result到正常异常的映射，下面将之映射为向量
from pandas import Series
lab_label=pd.read_csv('../data/feature/lab_label.csv')
#%%
lab_label=lab_label[['patientunitstayid','labresultoffset','labid','label']]
lab_label=lab_label.sort_values(by='patientunitstayid')
#%%
id=lab_label['labid'].drop_duplicates()
#%%
id=id.sort_values()
#%%
test=lab_label[:1000]
#%%
df_group = lab_label.groupby(['patientunitstayid','labresultoffset'])
# max_count_name =df_group.count().sort_values(by='labid',ascending=False)['NEW_CODE'].tolist()[0] #找到出现最多的有几个
new_df_colname = ['patientunitstayid','labresultoffset']+[i for i in id]  #生成新的df的列名
process_lab = pd.DataFrame(columns=new_df_colname)
#%%
num=len(df_group)
new_list=[]
with tqdm(total=num) as pbar:
    for i in df_group:
        info=[*list(i[0])]
        group_id=[*i[1]['labid'].tolist()]
        group_value=[*i[1]['label'].tolist()]
        # print(info)
        # data = [*list(i[0]), *i[1]['label'].tolist()]
        data={}
        data['patientunitstayid']=info[0]
        data['labresultoffset']=info[1]
        for i in range(len(group_id)):
            data[group_id[i]]=group_value[i]
        new_list.append(data)
        # ser = Series(data)
        # process_lab = process_lab.append(ser,ignore_index=True)
        # print(ser)
        # break
        pbar.update(1)
process_lab = process_lab.append(new_list)
#%%
test=process_lab[:1000]
#%%
process_lab.to_csv("../data/feature/process_lab.csv",index=False)
#%%
# process_lab 把1->-1,-1->1
process_lab=pd.read_csv("../data/feature/process_lab.csv")
#%%
#原0为正常，1为不正常，现将其进行调换
process_lab=process_lab.replace(1,8)
process_lab=process_lab.replace(0,1)
process_lab=process_lab.replace(8,0)
#%%
process_lab.to_csv("../data/feature/process_lab.csv",index=False)

#%%
patient=pd.read_csv("../data/feature/patient.csv")
#%%
patient420=patient[patient['hospitalid']==420]
#%%
patient420=patient420[['patientunitstayid', 'hospitalid']]
#%%
lab420=process_lab.merge(patient420,how='inner',on='patientunitstayid')
#%%
drug=pd.read_csv("../data/drug_time.csv")
#%%
drug420=drug.merge(patient420,how='inner',on='patientunitstayid')
#%%
lab420=lab420.drop('hospitalid',axis=1)
drug420=drug420.drop('hospitalid',axis=1)
#%%
row1=lab420[0:1]
row2=lab420[1:2]
#%%
lab420.to_csv('../data/feature/lab420_1130.csv',index=False)
drug420.to_csv('../data/feature/drug420_1130.csv',index=False)
#%%
lab420=pd.read_csv('../data/feature/lab420.csv')
drug420=pd.read_csv('../data/feature/drug420.csv')

#%%
lab_trans420=pd.read_csv('../data/feature/lab_trans420.csv')
#%%
transition_pd=pd.DataFrame(columns=lab420.columns)
transition_pd=transition_pd.drop(columns=['labresultoffset'])
#%%
lab_id=lab420['patientunitstayid'].drop_duplicates()
#%%
from tqdm import tqdm
with tqdm(total=len(lab_id)) as pbar:
    for i in lab_id:
        ori=lab420[lab420['patientunitstayid']==i]
        # print(i)
        trans=ori.shift(1)
        tmp_trans=ori-trans #求两行之差
        tmp_trans=tmp_trans[1:]
        tmp_trans=tmp_trans.drop(columns=['labresultoffset'])
        tmp_trans['patientunitstayid']=i
        transition_pd=transition_pd.append(tmp_trans)
        pbar.update(1)
        # break
#%%
transition_pd.to_csv('../data/feature/lab_trans420_1130.csv')
#%%
num_list=[]
for index,row in transition_pd.iterrows():
    num_data = {}
    row=row.drop('patientunitstayid')
    num=row.value_counts(normalize=True)
    if 1.0 in num:
        num_data['1']=num[1.0]
    else:
        num_data['1']=0
    if -1.0 in num:
        num_data['-1']=num[-1.0]
    else:
        num_data['-1']=0
    if 0.0 in num:
        num_data['0']=num[0.0]
    else:
        num_data['0']=0
    num_list.append(num_data)
    # break
#%%
lab_reward2 = pd.DataFrame(columns=['1','-1','0'])
#%%
lab_reward2 = lab_reward2.append(num_list)
#%%
lab_reward2.to_csv('../data/feature/lab_reward420.csv')

#%%
lab_id=lab420['patientunitstayid'].drop_duplicates()
lab_time=lab420[lab420['patientunitstayid']==3033830]['labresultoffset']
#%%
drug_id=drug['code'].drop_duplicates()
#%%
drug_id=drug_id.sort_values()
#%%
drug_action420=pd.DataFrame(columns=drug_id)
drug_action420.insert(0,'patientunitstayid',[])
drug_action420.insert(1,'time',[])
#%%
action_list=[]
with tqdm(total=len(lab_id)) as pbar:
    for i in lab_id:
        drug_i=drug420[drug420['patientunitstayid']==i]
        lab_time=lab420[lab420['patientunitstayid']==i]['labresultoffset']
        lab_time=lab_time.reset_index(drop=True)
        for j in range(0,len(lab_time)-1):
            time1=lab_time[j]
            time2=lab_time[j+1]
            tmp_drug={}
            tmp_drug['patientunitstayid']=i
            tmp_drug['time']=time1
            for index,row in drug_i.iterrows():
                if (row['drugstartoffset'] in range(time1,time2)) or (row['drugstopoffset'] in range(time1,time2)):
                    tmp_drug[row['code']]=1
                elif time1>=row['drugstartoffset'] and time2 <= row['drugstopoffset'] :
                    tmp_drug[row['code']]=1
                else:
                    continue
                # break
            action_list.append(tmp_drug)
            # break
        pbar.update(1)
#%%
drug_action420=drug_action420.append(action_list)
#%%
drug_action420=pd.read_csv("../data/feature/drug_action420.csv")
#%%
drug_action420=drug_action420.fillna(0)
#%%
drug_action420.to_csv("../data/feature/drug_action420.csv",index=False)
#%%
#-------------------graph_generation---------
drugname=pd.read_csv('../data/drugname.csv')
ddiname=pd.read_csv('../data/GRAPH/ddi_name.csv')
#%%
merge=drugname.merge(ddiname,left_on='drugname',right_on='drug_name')
#%%
#-----------筛选出本数据集出现的graph_id--------------
ddi=pd.read_csv('../data/GRAPH/ddi.csv')
#%%
drug_ddi_id=merge['DRUG_ID'].drop_duplicates()
#%%
drug_ddi_id=drug_ddi_id.reset_index(drop=True)
#%%
drop_index=[]
for index,row in ddi.iterrows():
    if row['id1'] not in drug_ddi_id.values or row['id2'] not in drug_ddi_id.values:
        drop_index.append(index)
#%%
drop_ddi=ddi.drop(index=drop_index)
#%%
test=drop_ddi['id1'].drop_duplicates()
test2=drop_ddi['id2'].drop_duplicates()
#%%
drug_id_dict={3:2}
for index, row in merge.iterrows():
    drug_id_dict[row['DRUG_ID']]=row['counts']
#%%
for index,row in drop_ddi.iterrows():
    id1=row['id1']
    id2=row['id2']
    if id1 in drug_id_dict:
        drop_ddi.at[index,'id1']=drug_id_dict[id1]
    if id2 in drug_id_dict:
        drop_ddi.at[index,'id2']=drug_id_dict[id2]
#%%
drop_ddi=drop_ddi.drop_duplicates()
#%%
drop_ddi.to_csv("../data/GRAPH/drop_ddi.csv",index=False)
#%%
patient=pd.read_csv('../data/feature/patient.csv')
patient420=pd.read_csv("../data/feature/patient420.csv")
#%%
patient420['admissionheight']=patient420['admissionheight'].fillna(round(patient420['admissionheight'].mean(),1))
patient420['admissionweight']=patient420['admissionweight'].fillna(round(patient420['admissionweight'].mean(),1))
#%%
patient420.to_csv("../data/feature/patient420.csv",index=False)
#%%
# 取医院420 的diagnose
patient420=pd.read_csv('../data/feature/patient420.csv')
diagnosis=pd.read_csv('../data/feature/diagnosis.csv')

#%%
patient420=patient420[['patientunitstayid','hospitalid']]
diagnosis=diagnosis.merge(patient420,on='patientunitstayid')
#%%
diagnosis=diagnosis.drop(['hospitalid'],axis=1)
#%%
diagnosis.to_csv('../data/feature/diagnosis420.csv',index=False)
#%%
lab_reward=pd.read_csv('../data/feature/lab_reward420.csv')
lab_reward=lab_reward.drop(['Unnamed: 0'],axis=1)
#%%
lab=pd.read_csv('../data/feature/lab420.csv')
#%%
lab_reward420=pd.DataFrame(columns=lab_reward.columns)
lab_reward420.insert(0,'patientunitstayid',[])
lab_reward420.insert(1,'time',[])
#%%
tmp_id=0
first_index=[]
for index, row in lab.iterrows():
    if row['patientunitstayid']==tmp_id:
        continue
    else:
        tmp_id=row['patientunitstayid']
        if(index!=0):
            first_index.append(index-1)
#%%
first_index.append(index)
#%%
lab=lab.drop(index=first_index)
#%%
pd.concat([df1, df2],axis=1)
#%%
lab=lab.reset_index(drop=True)
lab = lab[['patientunitstayid', 'labresultoffset']]
#%%
lab_reward420 = pd.concat([lab, lab_reward], axis=1)
#%%
lab_reward420.to_csv("../data/feature/lab_reward420.csv",index=False)
#%%
# 求四个数据的交集
patient = pd.read_csv('../data/feature/patient420.csv')
lab = pd.read_csv('../data/feature/lab420.csv')
diagnosis = pd.read_csv('../data/feature/diagnosis420.csv')
drug = pd.read_csv('../data/feature/drug_action420.csv')
#%%
id_dict={'3033830':1,}
patient_id=patient['patientunitstayid'].drop_duplicates()
lab_id=lab['patientunitstayid'].drop_duplicates()
diagnosis_id=diagnosis['patientunitstayid'].drop_duplicates()
drug_id=drug['patientunitstayid'].drop_duplicates()
#%%
for item in patient_id:
    id_dict[item]=1
#%%
for item in lab_id:
    id_dict[item]=id_dict.get(item, 0) + 1
#%%
for item in diagnosis_id:
    id_dict[item]=id_dict.get(item, 0) + 1
#%%
for item in drug_id:
    id_dict[item]=id_dict.get(item, 0) + 1
#%%
final_num=0
for value in id_dict.values():
    if value == 4:
        final_num=final_num+1


#%%
id_list=list(drug['patientunitstayid'].drop_duplicates())

#%%
drop_list=[]
diagnosis_id_list=list(diagnosis['patientunitstayid'].drop_duplicates())
for id in id_list:
    if id in diagnosis_id_list:
        drop_list.append(id)
#%%
id = drug[['patientunitstayid']].drop_duplicates()
#%%
id.to_csv("../data/feature/id_420.csv",index=False)
#%%
lab=pd.read_csv("../data/feature/lab420_1130.csv")
lab=lab.fillna(0)
#%%
lab.to_csv("../data/feature/lab420_126.csv",index=False)
#%%
times=lab['patientunitstayid'].value_counts()
#%%
patient = pd.read_csv("../data/patient.csv")
patient420 = pd.read_csv('../data/feature/patient420.csv')
patient=patient[['patientunitstayid', 'hospitaldischargestatus']]
test=patient420.merge(patient, on='patientunitstayid')
