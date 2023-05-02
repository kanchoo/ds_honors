#!/usr/bin/env python
# coding: utf-8

# # **The Impact of Data Scholars**
# 
# *Kanchana Samala*

# **Import Modules**

# In[621]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dataframe_image as dfi
import seaborn as sns 
from sklearn import metrics 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, r2_score


# **Load Datasets**

# In[622]:


# load all IUSE Data - Admin Datasets (appended cohort & term data)

admin_student = pd.read_csv('../_data_renamed/admin_appended/UCB_Admin_Student_FA2010FA2022.csv')
admin_course = pd.read_csv('../_data_renamed/admin_appended/UCB_Admin_Course_FA2010FA2022.csv') 

# load Fall 2021 & Spring 2022 IUSE Data - Survey Datasets

survey_pre_sp22 = pd.read_csv('../_data_renamed/survey_renamed/UCB_Survey_MainB_002_SP2022Pre.csv')
survey_post_sp22 = pd.read_csv('../_data_renamed/survey_renamed/UCB_Survey_MainB_003_SP2022Post.csv')

survey_pre_f21 = pd.read_csv('../_data_renamed/survey_renamed/UCB_Survey_MainA_007_FA2021Pre.csv')
survey_post_f21 = pd.read_csv('../_data_renamed/survey_renamed/UCB_Survey_MainB_001_FA2021Post.csv')

# load Data Scholars & Comparison Students Research IDs (Fall 2021 & Spring 2022)

scholars_f21 = pd.read_csv('../kanchana/dsus_data/f21_scholars_ids.csv')
control_f21 = pd.read_csv('../kanchana/dsus_data/f21_comparison_ids.csv')

scholars_sp22 = pd.read_csv('../kanchana/dsus_data/sp22_scholars_ids.csv')
control_sp22 = pd.read_csv('../kanchana/dsus_data/sp22_comparison_ids.csv')

print('F21 Scholars:', scholars_f21.shape, 'F21 Control:', control_f21.shape, 
      '\nSp22 Scholars:', scholars_sp22.shape, 'Sp22 Control:', control_sp22.shape)


# ***Performance Analysis***

# **Merge Admin & DSUS Datasets**

# In[623]:


# Merge IUSE Admin Datasets According to Control & Scholars Fall 2021 & Spring 2022

# Admin Course Data filtered for Data 8 Spring 2022 & Fall 2021

data_8_filter = admin_course['CourseNumber'] == 'C8'
fall_21_filter = admin_course['CourseYearTerm']=='2021 Fall'
spring_22_filter = admin_course['CourseYearTerm']=='2022 Spring'

f21_d8 = admin_course[data_8_filter & fall_21_filter]
sp22_d8 = admin_course[data_8_filter & spring_22_filter]

# Admin Course Data joined on Admin Student Data 

f21_admin = pd.merge(f21_d8, admin_student, on='ResearchID')
f21_admin['Data Scholars'] = f21_admin['ResearchID'].isin(scholars_f21['ResearchID'])
f21_admin['Control'] = f21_admin['ResearchID'].isin(control_f21['ResearchID'])

sp22_admin = pd.merge(sp22_d8, admin_student, on='ResearchID')
sp22_admin['Data Scholars'] = sp22_admin['ResearchID'].isin(scholars_sp22['ResearchID'])
sp22_admin['Control'] = sp22_admin['ResearchID'].isin(control_sp22['ResearchID'])

_admin = f21_admin.append(sp22_admin)

print('F21 Admin Merged:', f21_admin.shape, 'Sp22 Admin Merged:', sp22_admin.shape)
print('F21 Scholars Merged:', f21_admin[f21_admin['Data Scholars'] == True].shape, 
      'Sp22 Scholars Merged:', sp22_admin[sp22_admin['Data Scholars'] == True].shape)
print('Full Admin:', _admin.shape)


# **Slice Data**

# In[624]:


# Drop NA Grade Point Values & Add Pass, B or Better Columns

df = _admin.dropna(subset=['GradePointsNbr'])
print('Admin After Dropping NA Grade Point Values:', df.shape)
print('Scholars After Dropping NA Grade Point Values:', df[df['Data Scholars'] == True].shape)

df['Pass'] = df['GradePointsNbr'] >= 1.7
df['B or Better'] = df['GradePointsNbr'] >= 3.0

# Add Comparison Columns

df['Women vs. Men'] = df['PersonGenderDesc'] == 'Female'
df['Low SES vs. Not Low SES'] = df['LowSocioEconomicStatusFlg'] == 'Y'
df['FG vs. Not FG'] = df['FirstGenCollegeGradDesc'] == 'First Generation College'
df['Transfer vs. Not Transfer'] = df['ApplicantTypeCdShrtDesc'] == 'ADVANCED STANDING'


# In[625]:


# Filter Data for Gender (Female, Male) - Data Scholars only has Male & Female students

men_filter = df['PersonGenderDesc'] == 'Male'
women_filter = df['PersonGenderDesc'] == 'Female'

df = df[men_filter | women_filter]
print('\nAdmin After Filtering for Gender:', df.shape[0])
print('Scholars After Filtering for Gender:', df[df['Data Scholars'] == True].shape[0])

# Filter Data for College Generation Status (First, Not) 
# - Data Scholars only has 1 student with Unknown College Generation Status

fg_filter = df['FirstGenCollegeGradDesc'] == 'First Generation College'
non_fg_filter = df['FirstGenCollegeGradDesc'] == 'Not First Generation College'

df = df[fg_filter | non_fg_filter]
print('\nAdmin After Filtering for College Generation Status:', df.shape[0])
print('Scholars After Filtering for College Generation Status:', df[df['Data Scholars'] == True].shape[0])

# Filter Data for Ethnicity - Data Scholars has no Native American/Alaska Native or Pacific Islander students

na_filter = df['ShortEthnicDesc'] != 'Native American/Alaska Native'
pi_filter = df['ShortEthnicDesc'] != 'Pacific Islander'

df = df[na_filter & pi_filter]
print('\nAdmin After Race/Ethnicity Filter:', df.shape[0])
print('Scholars After Race/Ethnicity Filter:', df[df['Data Scholars'] == True].shape[0])


# **Make Descriptive Tables** 

# In[559]:


from IPython.core.display import display, HTML

def display_side_by_side(dfs:list, captions:list, tablespacing=5):
    """Display tables side by side to save vertical space
    Input:
        dfs: list of pandas.DataFrame
        captions: list of table captions
    """
    output = ""
    for (caption, df) in zip(captions, dfs):
        output += df.style.hide_index().set_table_attributes("style='display:inline'").set_caption(caption)._repr_html_()
        output += tablespacing * "\xa0"
    display(HTML(output))


# In[560]:


from IPython.display import Markdown, display
def printmd(string):
    display(Markdown(string))


# In[561]:


get_ipython().run_cell_magic('HTML', '', '<style type="text/css">\ntable.dataframe td, table.dataframe th {\n    border: 1px  black solid !important;\n  color: black !important;\n}\n</style>')


# **Justification for Scholars Analysis: Equity Gap Table**

# In[562]:


def equity_b_table(df, demo_filters, cols):
    b = df['GradePointsNbr'] >= 3.0
    _b = []
    _n = []
    for filt in demo_filters:
        n = df[filt].shape[0]
        _b.append(df[filt & b].shape[0] / n)
        _n.append(n)
    cols[0] = cols[0] + ' (n = ' + str(_n[0]) + ')'
    cols[1] = cols[1] + ' (n = ' + str(_n[1]) + ')'
    _b.append(_b[1] - _b[0])
    return pd.DataFrame(data=[_b], columns=cols)    


# In[563]:


def equity_pass_table(df, demo_filters, cols):
    b = df['GradePointsNbr'] >= 1.7
    _b = []
    _n = []
    for filt in demo_filters:
        n = df[filt].shape[0]
        _b.append(df[filt & b].shape[0] / n)
        _n.append(n)
    cols[0] = cols[0] + ' (n = ' + str(_n[0]) + ')'
    cols[1] = cols[1] + ' (n = ' + str(_n[1]) + ')'
    _b.append(_b[1] - _b[0])
    return pd.DataFrame(data=[_b], columns=cols)    


# In[564]:


d8_dfs = [equity_b_table(df, [df['UcbLevel1EthnicRollupDesc'] == 'Underrepresented Minority', 
                  df['UcbLevel1EthnicRollupDesc'] != 'Underrepresented Minority'],
            ['UR', 'non-UR', 'Gap']),
       
equity_b_table(df, [df['LowSocioEconomicStatusFlg'] == 'Y', 
                  df['LowSocioEconomicStatusFlg'] == 'N'], 
            ['Low SE', 'non-Low SE', 'Gap']),

equity_b_table(df, [df['FirstGenCollegeGradDesc'] == 'First Generation College', 
                  df['FirstGenCollegeGradDesc'] == 'Not First Generation College'],
            ['FG', 'Cont. G', 'Gap']),

equity_b_table(df, [df['ApplicantTypeCdShrtDesc'] == 'ADVANCED STANDING', 
                  df['ApplicantTypeCdShrtDesc'] == 'FRESHMAN, HS GRAD'], 
            ['Transfer', 'non-Transfer', 'Gap']),

equity_b_table(df, [df['PersonGenderDesc'] == 'Female', 
                  df['PersonGenderDesc'] == 'Male'], 
            ['Female', 'Male', 'Gap'])]


# In[565]:


d8_pass_dfs = [equity_pass_table(df, [df['UcbLevel1EthnicRollupDesc'] == 'Underrepresented Minority', 
                  df['UcbLevel1EthnicRollupDesc'] != 'Underrepresented Minority'],
            ['UR', 'non-UR', 'Gap']),
       
equity_pass_table(df, [df['LowSocioEconomicStatusFlg'] == 'Y', 
                  df['LowSocioEconomicStatusFlg'] == 'N'], 
            ['Low SE', 'non-Low SE', 'Gap']),

equity_pass_table(df, [df['FirstGenCollegeGradDesc'] == 'First Generation College', 
                  df['FirstGenCollegeGradDesc'] == 'Not First Generation College'],
            ['FG', 'Cont. G', 'Gap']),

equity_pass_table(df, [df['ApplicantTypeCdShrtDesc'] == 'ADVANCED STANDING', 
                  df['ApplicantTypeCdShrtDesc'] == 'FRESHMAN, HS GRAD'], 
            ['Transfer', 'non-Transfer', 'Gap']),

equity_pass_table(df, [df['PersonGenderDesc'] == 'Female', 
                  df['PersonGenderDesc'] == 'Male'], 
            ['Female', 'Male', 'Gap'])]


# In[566]:


scholars = df[df['Data Scholars'] == True]

scholars_dfs = [equity_b_table(scholars, [scholars['UcbLevel1EthnicRollupDesc'] == 'Underrepresented Minority', 
                  scholars['UcbLevel1EthnicRollupDesc'] != 'Underrepresented Minority'],
            ['UR', 'non-UR', 'Gap']),
       
equity_b_table(scholars, [scholars['LowSocioEconomicStatusFlg'] == 'Y', 
                  scholars['LowSocioEconomicStatusFlg'] == 'N'], 
            ['Low SE', 'non-Low SE', 'Gap']),

equity_b_table(scholars, [scholars['FirstGenCollegeGradDesc'] == 'First Generation College', 
                  scholars['FirstGenCollegeGradDesc'] == 'Not First Generation College'],
            ['FG', 'Cont. G', 'Gap']),

equity_b_table(scholars, [scholars['ApplicantTypeCdShrtDesc'] == 'ADVANCED STANDING', 
                  scholars['ApplicantTypeCdShrtDesc'] == 'FRESHMAN, HS GRAD'], 
            ['Transfer', 'non-Transfer', 'Gap']),

equity_b_table(scholars, [scholars['PersonGenderDesc'] == 'Female', 
                  scholars['PersonGenderDesc'] == 'Male'], 
            ['Female', 'Male', 'Gap'])]


# In[567]:


non_scholars = df[df['Data Scholars'] == False]

non_scholars_dfs = [equity_b_table(non_scholars, [non_scholars['UcbLevel1EthnicRollupDesc'] == 'Underrepresented Minority', 
                  non_scholars['UcbLevel1EthnicRollupDesc'] != 'Underrepresented Minority'],
            ['UR', 'non-UR', 'Gap']),
       
equity_b_table(non_scholars, [non_scholars['LowSocioEconomicStatusFlg'] == 'Y', 
                  non_scholars['LowSocioEconomicStatusFlg'] == 'N'], 
            ['Low SE', 'non-Low SE', 'Gap']),

equity_b_table(non_scholars, [non_scholars['FirstGenCollegeGradDesc'] == 'First Generation College', 
                  non_scholars['FirstGenCollegeGradDesc'] == 'Not First Generation College'],
            ['FG', 'Cont. G', 'Gap']),

equity_b_table(non_scholars, [non_scholars['ApplicantTypeCdShrtDesc'] == 'ADVANCED STANDING', 
                  non_scholars['ApplicantTypeCdShrtDesc'] == 'FRESHMAN, HS GRAD'], 
            ['Transfer', 'non-Transfer', 'Gap']),

equity_b_table(non_scholars, [non_scholars['PersonGenderDesc'] == 'Female', 
                  non_scholars['PersonGenderDesc'] == 'Male'], 
            ['Female', 'Male', 'Gap'])]


# In[568]:


print()
printmd('**Data 8**')
display_side_by_side(d8_dfs, ['% B or Better', '% B or Better', '% B or Better', '% B or Better', '% B or Better'],
                     tablespacing=5)
print()
printmd('**Data 8**')
display_side_by_side(d8_pass_dfs, ['Passing Rate', 'Passing Rate', 'Passing Rate', 'Passing Rate', 'Passing Rate'],
                     tablespacing=5)
print()
printmd('**Scholars**')
display_side_by_side(scholars_dfs, ['% B or Better', '% B or Better', '% B or Better', '% B or Better', '% B or Better'],
                     tablespacing=5)
print()
printmd('**non-Scholars**')
display_side_by_side(non_scholars_dfs, ['% B or Better', '% B or Better', '% B or Better', '% B or Better', '% B or Better'],
                     tablespacing=5)


# **Demographic Table of Student Participants in Admin/Performance Dataset**

# In[569]:


demo = pd.DataFrame()
demo[''] = ['African American/Black', 'Mexican American/Chicano', 'Other Hispanic/Latino', 'White',
                             'Chinese', 'Vietnamese', 'Korean', 'Filipino', 'Japanese', 'South Asian', 'Other Asian',
                             'International', 'Decline to State', 'Female', 'Male', 
                             'First Generation', 'Continuing Generation', 'Transfer', 'non-Transfer',
                             'Low Socioeconomic Status', 'non-Low Socioeconomic Status']
demo['Data Scholars (n=141)'] = ['6%', '33%', '12%', '9%',
                                     '4%', '6%', '1%', '5%', '1%', '9%', '4%', 
                                     '8%', '2%', '67%', '33%',
                                     '60%', '40%', '23%', '77%',
                                     '53%', '47%']
demo['non-Scholars (n=2526)'] = ['1%', '8%', '3%', '14%',
                                     '23%', '5%', '5%', '3%', '1%', '15%', '1%', 
                                     '17%', '4%', '52%', '48%',
                                     '25%', '75%', '14%', '86%',
                                     '19%', '81%']
demo['Total / Data 8 (n = 2667)'] = ['2%', '10%', '4%', '13%',
                                     '22%', '5%', '5%', '3%', '1%', '15%', '1%', 
                                     '16%', '3%', '52%', '48%',
                                     '26%', '74%', '14%', '86%',
                                     '21%', '79%']
demo.set_index('')


# **The Diverse Group of Students in Data Scholars**

# In[649]:


scholars = df['Data Scholars'] == True

urm = df[scholars]['UcbLevel1EthnicRollupDesc'].value_counts(normalize=True)['Underrepresented Minority'] 
low_ses = df[scholars]['LowSocioEconomicStatusFlg'].value_counts(normalize=True)['Y']
fem = df[scholars]['PersonGenderDesc'].value_counts(normalize=True)['Female']
fg = df[scholars]['FirstGenCollegeGradDesc'].value_counts(normalize=True)['First Generation College']

print('Proportion of URM Students in Data Scholars:', round(urm, 2))
print('Proportion of Low Socioeconomic Status Students in Data Scholars:', round(low_ses, 2))
print('Proportion of Female Students in Data Scholars:', round(fem, 2))
print('Proportion of First Generation Students in Data Scholars:', round(fg, 2))


# **Create Descriptive Tables regarding Student Performance**

# In[570]:


df[['Data Scholars', 'GradePointsNbr']].groupby(['Data Scholars']).mean()


# In[642]:


urm_ = df['UcbLevel1EthnicRollupDesc'] == 'Underrepresented Minority'
fg_ = df['FirstGenCollegeGradDesc'] == 'First Generation College'
ses_ = df['LowSocioEconomicStatusFlg'] == 'Y'
urm_fg_ses = df[urm_ & fg_ & ses_] # 175 students
urm_fg_ses[['Data Scholars', 'GradePointsNbr']].groupby(['Data Scholars']).mean()


# In[571]:


urm_fg_ses[['UcbLevel1EthnicRollupDesc', 'FirstGenCollegeGradDesc', 
        'LowSocioEconomicStatusFlg', 'ApplicantTypeCdShrtDesc', 'PersonGenderDesc', 'Data Scholars',
        'GradePointsNbr']].groupby(['UcbLevel1EthnicRollupDesc', 'FirstGenCollegeGradDesc', 
                                    'LowSocioEconomicStatusFlg', 'ApplicantTypeCdShrtDesc', 
                                    'PersonGenderDesc','Data Scholars']).mean()


# **Observe Different Associations Between Variables**

# In[572]:


# Correlation Between HS GPA & Data 8 GPA

scholars = df['Data Scholars'] == True
scholars_corr = df[scholars]['GpaHighSchoolUnweightedNbrFreshm'].corr(df[scholars]['GradePointsNbr'])

non_scholars = df['Data Scholars'] != True
non_corr = df[non_scholars]['GpaHighSchoolUnweightedNbrFreshm'].corr(df[non_scholars]['GradePointsNbr'])

print('Correlation Between HS GPA & Data 8 GPA')
print('Data Scholars (n='+ str(df[scholars]['GpaHighSchoolUnweightedNbrFreshm'].count()) +'):', scholars_corr)
print('Non-Scholars (n='+str( df[non_scholars]['GpaHighSchoolUnweightedNbrFreshm'].count())+'):', non_corr)


# In[573]:


# Correlation Between HS GPA & Data 8 GPA for URM students

urm = df['UcbLevel1EthnicRollupDesc'] == 'Underrepresented Minority'
scholars_corr = df[scholars & urm]['GpaHighSchoolUnweightedNbrFreshm'].corr(df[scholars & urm]['GradePointsNbr']) 
df[scholars & urm].shape[0]

non_corr = df[non_scholars & urm]['GpaHighSchoolUnweightedNbrFreshm'].corr(df[non_scholars & urm]['GradePointsNbr']) 
df[non_scholars & urm].shape[0]

print('Correlation Between HS GPA & Data 8 GPA for URM Students')
print('Data Scholars (n='+ str(df[scholars & urm]['GpaHighSchoolUnweightedNbrFreshm'].count()) +'):', scholars_corr)
print('Non-Scholars (n='+str( df[non_scholars & urm]['GpaHighSchoolUnweightedNbrFreshm'].count())+'):', non_corr)


# In[574]:


# Correlation between Transfer GPA & Data 8 GPA

transfer_s = df[scholars].dropna(subset=['GpaTransferNbrTransfer'])
scholars_corr = transfer_s['GpaTransferNbrTransfer'].corr(transfer_s['GradePointsNbr'])

transfer_n = df[non_scholars].dropna(subset=['GpaTransferNbrTransfer'])
non_corr = transfer_n['GpaTransferNbrTransfer'].corr(transfer_n['GradePointsNbr'])

print('Correlation Between Transfer GPA & Data 8 GPA')
print('Data Scholars (n='+ str(transfer_s.shape[0]) +'):', scholars_corr)
print('Non-Scholars (n='+str(transfer_n.shape[0])+'):', non_corr)


# In[575]:


# Correlation between Transfer GPA & Data 8 GPA for URM Students

transfer_s = df[scholars & urm].dropna(subset=['GpaTransferNbrTransfer'])
scholars_corr = transfer_s['GpaTransferNbrTransfer'].corr(transfer_s['GradePointsNbr'])

transfer_n = df[non_scholars & urm].dropna(subset=['GpaTransferNbrTransfer'])
non_corr = transfer_n['GpaTransferNbrTransfer'].corr(transfer_n['GradePointsNbr'])

print('Correlation Between Transfer GPA & Data 8 GPA for URM Students')
print('Data Scholars (n='+ str(transfer_s.shape[0]) +'):', scholars_corr)
print('Non-Scholars (n='+str(transfer_n.shape[0])+'):', non_corr)


# **Make Logistic Regression Model According to Admin Data with Focus on Performance: B or Better**

# Chosen Raw Features/Attributes: _PersonGenderDesc, UcbLevel1EthnicRollupDesc, ShortEthnicDesc, LowSocioEconomicStatusFlg, FirstGenCollegeGradDesc, ApplicantTypeCdShrtDesc, Data Scholars, GradePointsNbr_

# In[576]:


# Odds Ratio

print('Data Scholars vs. non-Scholars:', str(round(sum(df['Data Scholars'] == True) / df.shape[0], 3)))
print('Women vs. Men:', str(round(sum(df['PersonGenderDesc'] == 'Female') / df.shape[0], 3)))
print('Transfer vs. Freshman:', 
      str(round(sum(df['ApplicantTypeCdShrtDesc'] == 'ADVANCED STANDING') / df.shape[0], 3)))
print('FG vs. Cont. G:', 
      str(round(sum(df['FirstGenCollegeGradDesc'] == 'First Generation College') / df.shape[0], 3)))
print('Low SES vs. Not Low SES:', str(round(sum(df['LowSocioEconomicStatusFlg'] == 'Y') / df.shape[0], 3)))
print('African American/Black vs. White:', 
      str(round(sum(df['ShortEthnicDesc'] == 'African American/Black') / df.shape[0], 3)))
print('Mexican American/Chicano vs. White:', 
      str(round(sum(df['ShortEthnicDesc'] == 'Mexican American/Chicano') / df.shape[0], 3)))
print('Other Hispanic/Latino vs. White:', 
      str(round(sum(df['ShortEthnicDesc'] == 'Other Hispanic/Latino') / df.shape[0], 3)))
print('Korean vs. White:', 
      str(round(sum(df['ShortEthnicDesc'] == 'Korean') / df.shape[0], 3)))
print('Filipino vs. White:', 
      str(round(sum(df['ShortEthnicDesc'] == 'Filipino') / df.shape[0], 3)))
print('South Asian vs. White:', 
      str(round(sum(df['ShortEthnicDesc'] == 'South Asian') / df.shape[0], 3)))
print('Chinese vs. White:', 
      str(round(sum(df['ShortEthnicDesc'] == 'Chinese') / df.shape[0], 3)))
print('Vietnamese vs. White:', 
      str(round(sum(df['ShortEthnicDesc'] == 'Vietnamese') / df.shape[0], 3)))
print('Japanese vs. White:', 
      str(round(sum(df['ShortEthnicDesc'] == 'Japanese') / df.shape[0], 3)))
print('Other Asian vs. White:', 
      str(round(sum(df['ShortEthnicDesc'] == 'Other Asian') / df.shape[0], 3)))
print('International vs. White:', 
      str(round(sum(df['ShortEthnicDesc'] == 'International') / df.shape[0], 3)))
print('Decline to State Race vs. White:', 
      str(round(sum(df['ShortEthnicDesc'] == 'Decline to State') / df.shape[0], 3)))


# In[609]:


# Model According to Admin Data (Focus: Performance) Fall 2021 & Spring 2022

# X Columns of Interest: PersonGenderDesc -> Women vs. Men, 
#                        ShortEthnicDesc, 
#                        LowSocioEconomicStatusFlg -> Low SES vs. Not Low SES, 
#                        FirstGenCollegeGradDesc -> FG vs. Not FG, 
#                        ApplicantTypeCdShrtDesc -> Transfer vs. Not Tranfser, 
#                        Data Scholars
# Y Columns of Interest: 'GradePointsNbr'

x_df = df[['Data Scholars', 'Women vs. Men', 'ShortEthnicDesc', 'Low SES vs. Not Low SES', 
        'FG vs. Not FG', 'Transfer vs. Not Transfer']]

X = pd.get_dummies(x_df)
X = X[['Data Scholars', 'Women vs. Men', 'Low SES vs. Not Low SES', 
       'FG vs. Not FG', 'Transfer vs. Not Transfer', 
       'ShortEthnicDesc_African American/Black', 'ShortEthnicDesc_Mexican American/Chicano',
       'ShortEthnicDesc_Other Hispanic/Latino', 'ShortEthnicDesc_Korean', 'ShortEthnicDesc_Filipino',
       'ShortEthnicDesc_South Asian', 'ShortEthnicDesc_Chinese', 'ShortEthnicDesc_Vietnamese',
       'ShortEthnicDesc_Other Asian', 'ShortEthnicDesc_Japanese', 'ShortEthnicDesc_International', 
       'ShortEthnicDesc_Decline to State']]

y = df['B or Better']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

model = LogisticRegression(solver='liblinear', random_state=0)
model.fit(X_train, y_train)

printmd('Model Score: ' + str(round(model.score(X_test, y_test), 3)))
printmd('Mean Squared Error: ' + str(metrics.mean_squared_error(y_test.astype(np.float32), model.predict(X_test).astype(np.float32))))

columns = {'ShortEthnicDesc_African American/Black': 'African American/ Black', 
          'ShortEthnicDesc_Chinese': 'Chinese', 'ShortEthnicDesc_Decline to State': 'Decline to State Race',
          'ShortEthnicDesc_Filipino': 'Filipino', 'ShortEthnicDesc_International': 'Inter- national',
          'ShortEthnicDesc_Korean': 'Korean', 
          'ShortEthnicDesc_Mexican American/Chicano': 'Mexican American/ Chicano',
          'ShortEthnicDesc_Native American/Alaska Native': 'Native American/Alaska Native', 
          'ShortEthnicDesc_Other Hispanic/Latino': 'Other Hispanic/ Latino', 
          'ShortEthnicDesc_Other Asian': 'Other Asian', 'ShortEthnicDesc_Japanese': 'Japanese',
          'ShortEthnicDesc_South Asian': 'South Asian', 'ShortEthnicDesc_Vietnamese': 'Viet- namese'}

columns = {'ShortEthnicDesc_African American/Black': 'African American/ Black vs. White (0.015)', 
          'ShortEthnicDesc_Chinese': 'Chinese vs. White (0.216)', 
          'ShortEthnicDesc_Filipino': 'Filipino vs. White (0.028)', 
          'ShortEthnicDesc_Korean': 'Korean vs. White (0.05)', 
          'ShortEthnicDesc_Mexican American/Chicano': 'Mexican American/ Chicano vs. White (0.097)',
          'ShortEthnicDesc_Native American/Alaska Native': 'Native American/Alaska Native vs. White', 
          'ShortEthnicDesc_Other Hispanic/Latino': 'Other Hispanic/ Latino vs. White (0.035)',
          'ShortEthnicDesc_South Asian': 'South Asian vs. White (0.148)', 
          'ShortEthnicDesc_Vietnamese': 'Viet- namese vs. White (0.051)',
          'ShortEthnicDesc_Japanese': 'Japanese vs. White (0.012)',
          'ShortEthnicDesc_Other Asian': 'Other Asian vs. White (0.015)',
          'ShortEthnicDesc_International': 'Inter- national vs. White (0.162)',
          'ShortEthnicDesc_Decline to State': 'Decline to State Race vs. White (0.034)',
          'Data Scholars': 'Data Scholars vs. non- Scholars (0.053)',
          'Women vs. Men': 'Women vs. Men (0.525)', 'Other Asian': 'Other Asian (0.027)', 
          'Low SES vs. Not Low SES': 'Low SES vs. Not Low SES (0.211)', 'FG vs. Not FG': 'FG vs. Not FG (0.264)',
          'Transfer vs. Not Transfer': 'Transfer vs. Not Transfer (0.14)'}

pd.set_option('display.max_columns', None)
rounded_coef = [round(coef, 3) for coef in model.coef_[0]]
model_df = pd.DataFrame(data = [rounded_coef] , columns = X.columns)
model_df.rename(columns = columns)


# In[605]:


# Summary of Models According to Admin Data (Focus: Performance) Fall 2021 & Spring 2022

# X Columns of Interest: PersonGenderDesc, ShortEthnicDesc, LowSocioEconomicStatusFlg, 
#                        FirstGenCollegeGradDesc, 'ApplicantTypeCdShrtDesc', Data Scholars
# Y Columns of Interest: 'GradePointsNbr' -> 'B or Better'

x_df = df[['ShortEthnicDesc', 'Low SES vs. Not Low SES', 
        'FG vs. Not FG', 'Transfer vs. Not Transfer', 'Women vs. Men', 'Data Scholars']]

y = df['B or Better']

sum_df = pd.DataFrame()

columns = {'ShortEthnicDesc_African American/Black': 'African American/ Black vs. White (0.015)', 
          'ShortEthnicDesc_Chinese': 'Chinese vs. White (0.216)', 
          'ShortEthnicDesc_Filipino': 'Filipino vs. White (0.028)', 
          'ShortEthnicDesc_Korean': 'Korean vs. White (0.05)', 
          'ShortEthnicDesc_Mexican American/Chicano': 'Mexican American/ Chicano vs. White (0.097)',
          'ShortEthnicDesc_Native American/Alaska Native': 'Native American/Alaska Native vs. White', 
          'ShortEthnicDesc_Other Hispanic/Latino': 'Other Hispanic/ Latino vs. White (0.035)',
          'ShortEthnicDesc_South Asian': 'South Asian vs. White (0.148)', 
          'ShortEthnicDesc_Vietnamese': 'Viet- namese vs. White (0.051)',
          'ShortEthnicDesc_Japanese': 'Japanese vs. White (0.012)',
          'ShortEthnicDesc_Other Asian': 'Other Asian vs. White (0.015)',
          'ShortEthnicDesc_International': 'Inter- national vs. White (0.162)',
          'ShortEthnicDesc_Decline to State': 'Decline to State Race vs. White (0.034)',
          'Data Scholars': 'Data Scholars vs. non- Scholars (0.053)',
          'Women vs. Men': 'Women vs. Men (0.525)', 'Other Asian': 'Other Asian (0.027)', 
          'Low SES vs. Not Low SES': 'Low SES vs. Not Low SES (0.211)', 'FG vs. Not FG': 'FG vs. Not FG (0.264)',
          'Transfer vs. Not Transfer': 'Transfer vs. Not Transfer (0.14)'}

for i in np.arange(1, 64): # 2^6 - 1 possibilities for feature selection in Models
    a = np.array([[i]], dtype=np.uint8)
    X = pd.get_dummies(x_df.loc[:, np.unpackbits(a)[2:].astype(np.bool)])
    if 'ShortEthnicDesc_Korean' in X.columns:
        X = X.drop(columns=['ShortEthnicDesc_White'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    
    model = LogisticRegression(solver='liblinear', random_state=0)
    model.fit(X_train, y_train)
    
    rounded_coef = [round(coef, 3) for coef in model.coef_[0]]
    df2 = pd.DataFrame(data = [rounded_coef] , columns = X.columns)
    df2['Model Score'] = str(round(model.score(X_test, y_test), 3))
    df2['Mean Squared Error'] = metrics.mean_squared_error(y_test.astype(np.float32), 
                                                           model.predict(X_test).astype(np.float32))
    sum_df = pd.concat([sum_df, df2], axis = 0, ignore_index = True)

second = sum_df.pop('Mean Squared Error')
sum_df.insert(0, 'Mean Squared Error', second)
first = sum_df.pop('Model Score')
sum_df.insert(0, 'Model Score', first)
sum_df = sum_df.replace(np.nan,'',regex=True).rename(columns = columns)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

sum_df.to_csv('honors_models/performance_models.csv')
sum_df


# In[579]:


# Summary of Models According to Admin Data (Focus: Performance) Fall 2021 & Spring 2022

# X Columns of Interest: PersonGenderDesc, ShortEthnicDesc, LowSocioEconomicStatusFlg, 
#                        FirstGenCollegeGradDesc, 'ApplicantTypeCdShrtDesc', Data Scholars
# Y Columns of Interest: 'GradePointsNbr' -> 'B or Better'

x_df = df[['ShortEthnicDesc', 'Low SES vs. Not Low SES', 
        'FG vs. Not FG', 'Transfer vs. Not Transfer', 'Women vs. Men', 'Data Scholars']]

y = df['GradePointsNbr']

sum_df = pd.DataFrame()

columns = {'ShortEthnicDesc_African American/Black': 'African American/ Black vs. White (0.015)', 
          'ShortEthnicDesc_Chinese': 'Chinese vs. White (0.216)', 
          'ShortEthnicDesc_Filipino': 'Filipino vs. White (0.028)', 
          'ShortEthnicDesc_Korean': 'Korean vs. White (0.05)', 
          'ShortEthnicDesc_Mexican American/Chicano': 'Mexican American/ Chicano vs. White (0.097)',
          'ShortEthnicDesc_Native American/Alaska Native': 'Native American/Alaska Native vs. White', 
          'ShortEthnicDesc_Other Hispanic/Latino': 'Other Hispanic/ Latino vs. White (0.035)',
          'ShortEthnicDesc_South Asian': 'South Asian vs. White (0.148)', 
          'ShortEthnicDesc_Vietnamese': 'Viet- namese vs. White (0.051)',
          'ShortEthnicDesc_Japanese': 'Japanese vs. White (0.012)',
          'ShortEthnicDesc_Other Asian': 'Other Asian vs. White (0.015)',
          'ShortEthnicDesc_International': 'Inter- national vs. White (0.162)',
          'ShortEthnicDesc_Decline to State': 'Decline to State Race vs. White (0.034)',
          'Data Scholars': 'Data Scholars vs. non- Scholars (0.053)',
          'Women vs. Men': 'Women vs. Men (0.525)', 'Other Asian': 'Other Asian (0.027)', 
          'Low SES vs. Not Low SES': 'Low SES vs. Not Low SES (0.211)', 'FG vs. Not FG': 'FG vs. Not FG (0.264)',
          'Transfer vs. Not Transfer': 'Transfer vs. Not Transfer (0.14)'}

for i in np.arange(1, 64): # 2^6 - 1 possibilities for feature selection in Models
    a = np.array([[i]], dtype=np.uint8)
    X = pd.get_dummies(x_df.loc[:, np.unpackbits(a)[2:].astype(np.bool)])
    if 'ShortEthnicDesc_Korean' in X.columns:
        X = X.drop(columns=['ShortEthnicDesc_White'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    rounded_coef = [round(coef, 3) for coef in model.coef_]
    df2 = pd.DataFrame(data = [rounded_coef] , columns = X.columns)
    df2['Model Score'] = str(round(model.score(X_test, y_test), 3))
    df2['Mean Squared Error'] = metrics.mean_squared_error(y_test.astype(np.float32), 
                                                           model.predict(X_test).astype(np.float32)) 
    sum_df = pd.concat([sum_df, df2], axis = 0, ignore_index = True)

second = sum_df.pop('Mean Squared Error')
sum_df.insert(0, 'Mean Squared Error', second)
first = sum_df.pop('Model Score')
sum_df.insert(0, 'Model Score', first)
sum_df = sum_df.replace(np.nan,'',regex=True).rename(columns = columns)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#sum_df.to_csv('honors_models/performance_models.csv')
sum_df


# ***Belonging Analysis***

# **Merge Admin & DSUS & Survey Datasets**

# In[580]:


# Append Fall 2021 & Spring 2022 Survey Data
_survey_post = survey_post_f21.append(survey_post_sp22)

print('Raw Post Survey Size:', _survey_post.shape[0])

# Merge with Filtered Performance Admin Dataset
survey_post_admin = _survey_post.merge(df, on='ResearchID')

# Filter Surveys to 100% Filled out Responses
survey_post_admin = survey_post_admin[survey_post_admin['Progress'] == 100]

print('Filtered Post Survey Size:', survey_post_admin.shape[0], 
      'Filtered Post Survey Scholar Responses:', survey_post_admin[survey_post_admin['Data Scholars'] == True].shape[0])


# **Set up Survey Dataset for Modelling**

# Chosen Raw Features/Attributes: Q1_6, Q2_3_1, Q3_3_(1-4), Q4_4_(1-4), Q3_6_2, Q3_6_3, Q3_13(1, 3, 4, 5, 6)

# In[616]:


# Set up Maps for Question columns

neg = {'Frequently': 1, 'Sometimes': 2, 'Not at all': 5} #3_3, 3_4
pos = {'1= No, not at all': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7= Yes, very much': 7}
pos_agree = {'Strongly agree': 7, 'Agree': 6, 'Somewhat agree': 5, 'Neutral': 4, 'Neither agree nor disagree': 4, 
             'Somewhat disagree': 3, 'Disagree': 2, 'Strongly disagree': 1}
neg_agree = {'Strongly agree': 1, 'Agree': 2, 'Somewhat agree': 3, 'Neutral': 4, 'Neither agree nor disagree': 4, 
             'Somewhat disagree': 5, 'Disagree': 6, 'Strongly disagree': 7}

# Set up X columns for models

survey_post_admin['Prior CS Expo- sure'] = survey_post_admin['Q1_6'] == 'Yes'

# Set up y column for models

y_1 = survey_post_admin['Q2_3_1'].map(pos) 
# I see myself as a data science person. 
y_2 = survey_post_admin['Q3_3_1'].map(neg) 
# I have been insulted or threatened by other students in a data science class because of my race/ethnicity.
y_3 = survey_post_admin['Q3_3_2'].map(neg) 
# I have heard uGSIs in a data science class make inappropriate remarks regarding racial/ethnic minorities.
y_4 = survey_post_admin['Q3_3_3'].map(neg)
# I have heard instructors in a data science class make inappropriate remarks regarding racial/ethnic minorities.
y_5 = survey_post_admin['Q3_3_4'].map(neg)
# I have felt excluded from activities in a data science class because of my race/ethnicity.
y_6 = survey_post_admin['Q3_4_1'].map(neg)
# I have been insulted or threatened by other students in a data science class because of my gender. 
y_7 = survey_post_admin['Q3_4_2'].map(neg)
# I have heard uGSIs in a data science class make inappropriate remarks regarding woman-identifying students. 
y_8 = survey_post_admin['Q3_4_3'].map(neg)
# I have heard instructors in a data science class make inappropriate remarks regarding woman-identifying students.
y_9 = survey_post_admin['Q3_4_4'].map(neg)
# I have felt excluded from activities in a data science class because of my gender.
y_10 = survey_post_admin['Q3_13_1'].map(pos_agree)
# When collaborating with my fellow peers in Data 8, I feel like my perspective is valuable.
y_11 = survey_post_admin['Q3_13_3'].map(pos_agree)
# I feel comfortable collaborating with most of my fellow peers in Data 8.
y_12 = survey_post_admin['Q3_13_4'].map(pos_agree)
# My classroom experiences with fellow students in Data 8 has been mostly positive.
y_13 = survey_post_admin['Q3_13_5'].map(pos_agree)
# My fellow peers generally make space for me to contribute to class discussions. 
y_14 = survey_post_admin['Q3_13_6'].map(neg_agree)
# Classroom experiences with fellow students in Data 8 have made me feel like I don't belong in data science.
y_15 = survey_post_admin['Q3_6_2'].map(neg_agree)
# When something bad happens, I feel that maybe I don’t belong in data science
y_16 = survey_post_admin['Q3_6_3'].map(pos_agree)
# When something good happens, I feel that I really belong in data science.

survey_post_admin['Sense of Belonging'] = (y_1 + y_2 + y_3 + y_4 + y_5 + y_6 + y_7 + y_8 + y_9 + 
                          y_10 + y_11 + y_12 + y_13 + y_14 + y_15 + y_16) / 16
survey = survey_post_admin[pd.notna(survey_post_admin['Sense of Belonging'])]

print('Post Survey Size:', survey.shape[0], 
      'Post Survey Scholar Responses:', survey[survey['Data Scholars'] == True].shape[0])


# **Demographic of Student Participants**

# In[582]:


demo = pd.DataFrame()
demo[''] = ['African American/Black', 'Mexican American/Chicano', 'Other Hispanic/Latino', 'White',
                             'Chinese', 'Vietnamese', 'Korean', 'Filipino', 'Japanese', 'South Asian', 'Other Asian',
                             'International', 'Decline to State', 'Female', 'Male', 
                             'First Generation', 'Continuing Generation', 'Transfer', 'non-Transfer',
                             'Low Socioeconomic Status', 'non-Low Socioeconomic Status']
demo['Data Scholars (n=23)'] = ['6%', '33%', '12%', '9%',
                                     '4%', '6%', '1%', '5%', '1%', '9%', '4%', 
                                     '8%', '2%', '65%', '35%',
                                     '74%', '26%', '26%', '74%',
                                     '59%', '41%']
demo['non-Scholars (n=187)'] = ['1%', '6%', '5%', '13%',
                                     '33%', '7%', '3%', '3%', '2%', '8%', '2%', 
                                     '11%', '5%', '63%', '37%',
                                     '26%', '74%', '6%', '94%',
                                     '21%', '79%']
demo['Total / Data 8 (n = 210)'] = ['1%', '9%', '6%', '13%',
                                     '30%', '9%', '3%', '2%', '2%', '8%', '2%', 
                                     '10%', '5%', '63%', '37%',
                                     '31%', '69%', '8%', '92%',
                                     '25%', '75%']
demo.set_index('')


# **Create Descriptive Tables regarding Student Sense of Belonging**

# In[643]:


survey[['Sense of Belonging', 'Data Scholars']].groupby(['Data Scholars']).mean()


# In[645]:


urm_ = survey['UcbLevel1EthnicRollupDesc'] == 'Underrepresented Minority'
fg_ = survey['FirstGenCollegeGradDesc'] == 'First Generation College'
ses_ = survey['LowSocioEconomicStatusFlg'] == 'Y'
urm_fg_ses = survey[urm_ & fg_ & ses_] # 19 students
urm_fg_ses[['Data Scholars', 'Sense of Belonging']].groupby(['Data Scholars']).mean()


# In[644]:


urm_fg_ses[['UcbLevel1EthnicRollupDesc', 'FirstGenCollegeGradDesc', 
        'LowSocioEconomicStatusFlg', 'ApplicantTypeCdShrtDesc', 'PersonGenderDesc', 'Data Scholars',
        'Sense of Belonging']].groupby(['UcbLevel1EthnicRollupDesc', 'FirstGenCollegeGradDesc', 
                                    'LowSocioEconomicStatusFlg', 'ApplicantTypeCdShrtDesc', 
                                    'PersonGenderDesc','Data Scholars']).mean()


# **Observe Different Associations Between Variables**

# In[619]:


# Correlation Between HS GPA & Data 8 GPA

scholars = survey['Data Scholars'] == True
scholars_corr = survey[scholars]['Sense of Belonging'].corr(survey[scholars]['GradePointsNbr'])

non_scholars = survey['Data Scholars'] != True
non_corr = survey[non_scholars]['Sense of Belonging'].corr(survey[non_scholars]['GradePointsNbr'])

print('Correlation Between Data 8 GPA & Sense of Belonging')
print('Data Scholars (n='+ str(survey[scholars].shape[0]) +'):', scholars_corr)
print('Non-Scholars (n='+str( survey[non_scholars].shape[0])+'):', non_corr)


# In[620]:


# Correlation Between HS GPA & Data 8 GPA for URM students

urm = survey['UcbLevel1EthnicRollupDesc'] == 'Underrepresented Minority'
scholars_corr = survey[scholars & urm]['Sense of Belonging'].corr(survey[scholars]['GradePointsNbr'])

non_corr = survey[non_scholars & urm]['Sense of Belonging'].corr(survey[non_scholars]['GradePointsNbr'])

print('Correlation Between Data 8 GPA & Sense of Belonging for URM Students')
print('Data Scholars (n='+ str(survey[scholars & urm].shape[0]) +'):', scholars_corr)
print('Non-Scholars (n='+str( survey[non_scholars & urm].shape[0])+'):', non_corr)


# **Make Logistic Regression Model According to Admin Data with Focus on Sense of Belonging: > 4**

# Chosen Raw Features/Attributes: PersonGenderDesc, UcbLevel1EthnicRollupDesc, ShortEthnicDesc, LowSocioEconomicStatusFlg, FirstGenCollegeGradDesc, ApplicantTypeCdShrtDesc, Data Scholars, GradePointsNbr

# In[598]:


# Odds Ratio

print('Data Scholars vs. non-Scholars:', str(round(sum(survey['Data Scholars'] == True) / survey.shape[0], 3)))
print('Women vs. Men:', str(round(sum(survey['PersonGenderDesc'] == 'Female') / survey.shape[0], 3)))
print('Transfer vs. Freshman:', 
      str(round(sum(survey['ApplicantTypeCdShrtDesc'] == 'ADVANCED STANDING') / survey.shape[0], 3)))
print('FG vs. Cont. G:', 
      str(round(sum(survey['FirstGenCollegeGradDesc'] == 'First Generation College') / survey.shape[0], 3)))
print('Low SES vs. Not Low SES:', str(round(sum(survey['LowSocioEconomicStatusFlg'] == 'Y') / survey.shape[0], 3)))
print('African American/Black vs. White:', 
      str(round(sum(survey['ShortEthnicDesc'] == 'African American/Black') / survey.shape[0], 3)))
print('Mexican American/Chicano vs. White:', 
      str(round(sum(survey['ShortEthnicDesc'] == 'Mexican American/Chicano') / survey.shape[0], 3)))
print('Other Hispanic/Latino vs. White:', 
      str(round(sum(survey['ShortEthnicDesc'] == 'Other Hispanic/Latino') / survey.shape[0], 3)))
print('Korean vs. White:', 
      str(round(sum(survey['ShortEthnicDesc'] == 'Korean') / survey.shape[0], 3)))
print('Filipino vs. White:', 
      str(round(sum(survey['ShortEthnicDesc'] == 'Filipino') / survey.shape[0], 3)))
print('South Asian vs. White:', 
      str(round(sum(survey['ShortEthnicDesc'] == 'South Asian') / survey.shape[0], 3)))
print('Chinese vs. White:', 
      str(round(sum(survey['ShortEthnicDesc'] == 'Chinese') / survey.shape[0], 3)))
print('Vietnamese vs. White:', 
      str(round(sum(survey['ShortEthnicDesc'] == 'Vietnamese') / survey.shape[0], 3)))
print('Japanese vs. White:', 
      str(round(sum(survey['ShortEthnicDesc'] == 'Japanese') / survey.shape[0], 3)))
print('Other Asian vs. White:', 
      str(round((sum(survey['ShortEthnicDesc'] == 'Other Asian')) / survey.shape[0], 3)))
print('International vs. White:', 
      str(round(sum(survey['ShortEthnicDesc'] == 'International') / survey.shape[0], 3)))
print('Decline to State Race vs. White:', 
      str(round((sum(survey['ShortEthnicDesc'] == 'Decline to State')) / survey.shape[0], 3)))


# In[610]:


# Model According to Admin Data (Focus: Belonging) Fall 2021 & Spring 2022

# X Columns of Interest: PersonGenderDesc, ShortEthnicDesc, LowSocioEconomicStatusFlg, 
#                        FirstGenCollegeGradDesc, 'ApplicantTypeCdSh\rtDesc', Data Scholars
# Y Columns of Interest: 'GradePointsNbr'

x_df = survey[['Data Scholars', 'Women vs. Men', 'ShortEthnicDesc', 'Low SES vs. Not Low SES', 
        'FG vs. Not FG', 'Transfer vs. Not Transfer', 'Prior CS Expo- sure', 'B or Better']]

X = pd.get_dummies(x_df)
X = X[['Data Scholars', 'Women vs. Men', 'Low SES vs. Not Low SES', 
       'FG vs. Not FG', 'Transfer vs. Not Transfer', 'Prior CS Expo- sure', 
       'ShortEthnicDesc_African American/Black', 'ShortEthnicDesc_Mexican American/Chicano',
       'ShortEthnicDesc_Other Hispanic/Latino', 'ShortEthnicDesc_Korean', 'ShortEthnicDesc_Filipino',
       'ShortEthnicDesc_South Asian', 'ShortEthnicDesc_Chinese', 'ShortEthnicDesc_Vietnamese',
       'ShortEthnicDesc_Other Asian', 'ShortEthnicDesc_Japanese', 'ShortEthnicDesc_International', 
       'ShortEthnicDesc_Decline to State','B or Better']]

y = survey['Sense of Belonging'] > 4

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

model = LogisticRegression(solver='liblinear', random_state=0)
model.fit(X_train, y_train)

printmd('Model Score: ' + str(round(model.score(X_test, y_test), 3)))
printmd('Mean Squared Error: ' + str(round(metrics.mean_squared_error(y_test.astype(np.float32), 
                                                           model.predict(X_test).astype(np.float32)), 3)))

columns = {'ShortEthnicDesc_African American/Black': 'African American/ Black', 
          'ShortEthnicDesc_Chinese': 'Chinese', 'ShortEthnicDesc_Decline to State': 'Decline to State Race',
          'ShortEthnicDesc_Filipino': 'Filipino', 'ShortEthnicDesc_International': 'Inter- national',
          'ShortEthnicDesc_Korean': 'Korean', 
          'ShortEthnicDesc_Mexican American/Chicano': 'Mexican American/ Chicano',
          'ShortEthnicDesc_Native American/Alaska Native': 'Native American/Alaska Native', 
          'ShortEthnicDesc_Other Hispanic/Latino': 'Other Hispanic/ Latino',
          'ShortEthnicDesc_Other Asian': 'Other Asian', 'ShortEthnicDesc_Japanese': 'Japa- nese',
          'ShortEthnicDesc_South Asian': 'South Asian', 'ShortEthnicDesc_Vietnamese': 'Viet- namese'}

columns = {'ShortEthnicDesc_African American/Black': 'African American/ Black vs. White (0.01)', 
          'ShortEthnicDesc_Chinese': 'Chinese vs. White (0.305)', 
          'ShortEthnicDesc_Filipino': 'Filipino vs. White (0.024)', 
          'ShortEthnicDesc_International': 'Inter- national vs. White (0.105)',
          'ShortEthnicDesc_Korean': 'Korean vs. White (0.029)', 
          'ShortEthnicDesc_Mexican American/Chicano': 'Mexican American/ Chicano vs. White (0.09)',
          'ShortEthnicDesc_Other Hispanic/Latino': 'Other Hispanic/ Latino vs. White (0.057)',
          'ShortEthnicDesc_South Asian': 'South Asian vs. White (0.081)', 
          'ShortEthnicDesc_Vietnamese': 'Viet- namese vs. White (0.086)',
          'ShortEthnicDesc_Japanese': 'Japa- nese vs. White (0.019)',
          'ShortEthnicDesc_Other Asian': 'Other Asian vs. White (0.019)',
          'ShortEthnicDesc_Decline to State': 'Decline to State Race (0.048)',
          'Data Scholars': 'Data Scholars vs. non- Scholars (0.11)',
          'Women vs. Men': 'Women vs. Men (0.629)', 'Other Asian': 'Other Asian (0.038)', 
          'Low SES vs. Not Low SES': 'Low SES vs. Not Low SES (0.248)', 'FG vs. Not FG': 'FG vs. Not FG (0.314)',
          'Transfer vs. Not Transfer': 'Transfer vs. Not Transfer (0.081)'}

pd.set_option('display.max_columns', None)
rounded_coef = [round(coef, 3) for coef in model.coef_[0]]
model_df = pd.DataFrame(data = [rounded_coef] , columns = X.columns)
model_df.rename(columns = columns)


# In[604]:


# Summary of Models According to Admin Data (Focus: Belonging) Fall 2021 & Spring 2022

# X Columns of Interest: PersonGenderDesc, ShortEthnicDesc, LowSocioEconomicStatusFlg, 
#                        FirstGenCollegeGradDesc, 'ApplicantTypeCdShrtDesc', Data Scholars
# Y Columns of Interest: 'Sense of Belonging' > 4


x_df = survey[['ShortEthnicDesc', 'Low SES vs. Not Low SES', 'Prior CS Expo- sure',
        'FG vs. Not FG', 'Transfer vs. Not Transfer', 'Women vs. Men', 'B or Better', 'Data Scholars']]

y = survey['Sense of Belonging'] > 4

sum_df = pd.DataFrame()

columns = {'ShortEthnicDesc_African American/Black': 'African American/ Black vs. White (0.01)', 
          'ShortEthnicDesc_Chinese': 'Chinese vs. White (0.305)', 
          'ShortEthnicDesc_Filipino': 'Filipino vs. White (0.024)', 
          'ShortEthnicDesc_International': 'Inter- national vs. White (0.105)',
          'ShortEthnicDesc_Korean': 'Korean vs. White (0.029)', 
          'ShortEthnicDesc_Mexican American/Chicano': 'Mexican American/ Chicano vs. White (0.09)',
          'ShortEthnicDesc_Other Hispanic/Latino': 'Other Hispanic/ Latino vs. White (0.057)',
          'ShortEthnicDesc_South Asian': 'South Asian vs. White (0.081)', 
          'ShortEthnicDesc_Vietnamese': 'Viet- namese vs. White (0.086)',
          'ShortEthnicDesc_Japanese': 'Japa- nese vs. White (0.019)',
          'ShortEthnicDesc_Other Asian': 'Other Asian vs. White (0.019)',
          'ShortEthnicDesc_Decline to State': 'Decline to State Race (0.048)',
          'Data Scholars': 'Data Scholars vs. non- Scholars (0.11)',
          'Women vs. Men': 'Women vs. Men (0.629)', 'Other Asian': 'Other Asian (0.038)', 
          'Low SES vs. Not Low SES': 'Low SES vs. Not Low SES (0.248)', 'FG vs. Not FG': 'FG vs. Not FG (0.314)',
          'Transfer vs. Not Transfer': 'Transfer vs. Not Transfer (0.081)'}

for i in np.arange(1, 256): # 2^8 - 1 possibilities for feature selection in Models
    a = np.array([[i]], dtype=np.uint8)
    X = pd.get_dummies(x_df.loc[:, np.unpackbits(a).astype(np.bool)])
    if 'ShortEthnicDesc_Japanese' in X.columns:
        X = X.drop(columns=['ShortEthnicDesc_White'])
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    
    model = LogisticRegression(solver='liblinear', random_state=0)
    model.fit(X_train, y_train)
    
    rounded_coef = [round(coef, 3) for coef in model.coef_[0]]
    df2 = pd.DataFrame(data = [rounded_coef] , columns = X.columns)
    df2['Mean Squared Error'] = metrics.mean_squared_error(y_test.astype(np.float32), 
                                                           model.predict(X_test).astype(np.float32)) 
    df2['Model Score'] = model.score(X_test, y_test)
    
    #df2['r2'] = r2_score(y.astype(np.float32), model.predict(X).astype(np.float32)) 
    sum_df = pd.concat([sum_df, df2], axis = 0, ignore_index = True)

first = sum_df.pop('Model Score')
sum_df.insert(0, 'Model Score', first)
sum_df = sum_df.replace(np.nan,'',regex=True).rename(columns = columns)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

sum_df.to_csv('honors_models/belonging_models.csv')
sum_df


# In[601]:


# Summary of Models According to Admin Data (Focus: Belonging) Fall 2021 & Spring 2022

# X Columns of Interest: PersonGenderDesc, ShortEthnicDesc, LowSocioEconomicStatusFlg, 
#                        FirstGenCollegeGradDesc, 'ApplicantTypeCdShrtDesc', Data Scholars
# Y Columns of Interest: 'Sense of Belonging' > 4


x_df = survey[['ShortEthnicDesc', 'Low SES vs. Not Low SES', 'Prior CS Expo- sure',
        'FG vs. Not FG', 'Transfer vs. Not Transfer', 'Women vs. Men', 'B or Better', 'Data Scholars']]

y = survey['Sense of Belonging'] > 4

sum_df = pd.DataFrame()

columns = {'ShortEthnicDesc_African American/Black': 'African American/ Black vs. White (0.01)', 
          'ShortEthnicDesc_Chinese': 'Chinese vs. White (0.305)', 
          'ShortEthnicDesc_Filipino': 'Filipino vs. White (0.024)', 
          'ShortEthnicDesc_International': 'Inter- national vs. White (0.105)',
          'ShortEthnicDesc_Korean': 'Korean vs. White (0.029)', 
          'ShortEthnicDesc_Mexican American/Chicano': 'Mexican American/ Chicano vs. White (0.09)',
          'ShortEthnicDesc_Other Hispanic/Latino': 'Other Hispanic/ Latino vs. White (0.057)',
          'ShortEthnicDesc_South Asian': 'South Asian vs. White (0.081)', 
          'ShortEthnicDesc_Vietnamese': 'Viet- namese vs. White (0.086)',
          'ShortEthnicDesc_Japanese': 'Japa- nese vs. White (0.019)',
          'ShortEthnicDesc_Other Asian': 'Other Asian vs. White (0.019)',
          'ShortEthnicDesc_Decline to State': 'Decline to State Race (0.048)',
          'Data Scholars': 'Data Scholars vs. non- Scholars (0.11)',
          'Women vs. Men': 'Women vs. Men (0.629)', 'Other Asian': 'Other Asian (0.038)', 
          'Low SES vs. Not Low SES': 'Low SES vs. Not Low SES (0.248)', 'FG vs. Not FG': 'FG vs. Not FG (0.314)',
          'Transfer vs. Not Transfer': 'Transfer vs. Not Transfer (0.081)'}

for i in np.arange(1, 256): # 2^8 - 1 possibilities for feature selection in Models
    a = np.array([[i]], dtype=np.uint8)
    X = pd.get_dummies(x_df.loc[:, np.unpackbits(a).astype(np.bool)])
    if 'ShortEthnicDesc_Japanese' in X.columns:
        X = X.drop(columns=['ShortEthnicDesc_White'])
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    rounded_coef = [round(coef, 3) for coef in model.coef_]
    df2 = pd.DataFrame(data = [rounded_coef] , columns = X.columns)
    df2['Mean Squared Error'] = metrics.mean_squared_error(y_test.astype(np.float32), 
                                                           model.predict(X_test).astype(np.float32)) 
    df2['Model Score'] = model.score(X_test, y_test)
    
    #df2['r2'] = r2_score(y.astype(np.float32), model.predict(X).astype(np.float32)) 
    sum_df = pd.concat([sum_df, df2], axis = 0, ignore_index = True)

first = sum_df.pop('Model Score')
sum_df.insert(0, 'Model Score', first)
sum_df = sum_df.replace(np.nan,'',regex=True).rename(columns = columns)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

sum_df#.to_csv('Belonging Models.csv')
sum_df

