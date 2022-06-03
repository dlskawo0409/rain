#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.metrics as mt
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier


# In[3]:


weather_df=pd.read_csv("./data/pre_labeled_weather.csv", index_col = 0)


# In[4]:


weather_df


# In[5]:


Y = weather_df.TomorrowRain
X = weather_df.drop("TomorrowRain",axis=1)


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2,stratify = Y, random_state = 0)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
e
print(X_train.shape ,X_val.shape, X_test.shape)


# In[7]:


def get_clf_eval(y_test, pred=None):
    confusion = mt.confusion_matrix( y_test, pred)
    accuracy = mt.accuracy_score(y_test , pred)
    precision = mt.precision_score(y_test , pred)
    recall = mt.recall_score(y_test , pred)
    f1 = mt.f1_score(y_test,pred)

    print('오차 행렬:')
    print(confusion)
    
    print('\n정확도: {0:.4f} \n정밀도: {1:.4f} \n재현율: {2:.4f} \n    F1: {3:.4f}'.format(accuracy, precision, recall, f1))


# In[19]:


lgbm_clf = LGBMClassifier(n_estimators=2000, num_leaves=95, subsample=0.8, min_child_samples=100,learning_rate = 0.12,
                          max_depth=128)

evals = [(X_train, y_train),(X_val, y_val)]
lgbm_clf.fit(X_train, y_train, early_stopping_rounds=100, eval_metric="accuracy", eval_set=evals,
                verbose=True)

preds = lgbm_clf.predict(X_test)
get_clf_eval(y_test, preds)


# In[21]:


lgbm_clf = LGBMClassifier(n_estimators=2000, num_leaves=100, subsample=0.6, min_child_samples=100,learning_rate = 0.12,
                          max_depth=128,col)

evals = [(X_train, y_train),(X_val, y_val)]
lgbm_clf.fit(X_train, y_train, early_stopping_rounds=100, eval_metric="accuracy", eval_set=evals,
                verbose=True)

preds = lgbm_clf.predict(X_test)
get_clf_eval(y_test, preds)


# In[24]:


lgbm_clf = LGBMClassifier(n_estimators=2500, num_leaves=100, subsample=0.6, min_child_samples=110,learning_rate = 0.09,
                          max_depth=148,colsample_bytree= 0.8)

evals = [(X_train, y_train),(X_val, y_val)]
lgbm_clf.fit(X_train, y_train, early_stopping_rounds=100, eval_metric="accuracy", eval_set=evals,
                verbose=True)

preds = lgbm_clf.predict(X_test)
get_clf_eval(y_test, preds)


# In[9]:


lgbm_clf = LGBMClassifier(n_estimators=2000, num_leaves=100, subsample=0.6, min_child_samples=80,learning_rate = 0.12,
                          max_depth=128,colsample_bytree= 0.8)

evals = [(X_train, y_train),(X_val, y_val)]
lgbm_clf.fit(X_train, y_train, early_stopping_rounds=100, eval_metric="accuracy", eval_set=evals,
                verbose=True)

preds = lgbm_clf.predict(X_test)
get_clf_eval(y_test, preds)


# In[ ]:





# In[ ]:





# In[5]:


from sklearn.model_selection import GridSearchCV

# 하이퍼 파라미터 테스트의 수행 속도를 향상시키기 위해 n_estimators를 100으로 감소
lgbm_wrapper = LGBMClassifier(n_estimators=200)

params = {'learnig_rate':[0.1, 0.5,0.8] ,'max_depth': [3,5,7,128],'num_leaves':[31,15,64],'min_child_samples':[60,100],
          'subsample': [0.8,1]}

# cv는 3으로 지정 
gridcv = GridSearchCV(lgbm_wrapper, param_grid=params, cv=2)
gridcv.fit(X_train, y_train, early_stopping_rounds=30, eval_metric="error",
           eval_set=[(X_val,y_val)])

print('GridSearchCV 최적 파라미터:',gridcv.best_params_) 

xgb_roc_score = roc_auc_score(y_test, gridcv.predict_proba(X_test)[:,1], average='macro')
print('ROC AUC: {0:.4f}'.format(xgb_roc_score))


# In[12]:


lgbm_wrapper = LGBMClassifier(n_estimators=200,max_depth=128,min_child_samples=60,num_leaves=64,subsample=0.8)

# LightGBM도 XGBoost와 동일하게 조기 중단 수행 가능. 
evals=[(X_val,y_val)]
lgbm_wrapper.fit(X_train, y_train, early_stopping_rounds=100, eval_metric="logloss", 
                 eval_set=evals, verbose=True)
preds = lgbm_wrapper.predict(X_test)
pred_proba = lgbm_wrapper.predict_proba(X_test)[:, 1]
get_clf_eval(y_test, preds, pred_proba)


# In[13]:


lgbm_wrapper = LGBMClassifier(n_estimators=400,max_depth=128,min_child_samples=60,num_leaves=64,subsample=0.8)

# LightGBM도 XGBoost와 동일하게 조기 중단 수행 가능. 
evals=[(X_val,y_val)]
lgbm_wrapper.fit(X_train, y_train, early_stopping_rounds=100, eval_metric="logloss", 
                 eval_set=evals, verbose=True)
preds = lgbm_wrapper.predict(X_test)
pred_proba = lgbm_wrapper.predict_proba(X_test)[:, 1]
get_clf_eval(y_test, preds, pred_proba)


# In[14]:


lgbm_wrapper = LGBMClassifier(n_estimators=800,max_depth=128,min_child_samples=60,num_leaves=64,subsample=0.8)

# LightGBM도 XGBoost와 동일하게 조기 중단 수행 가능. 
evals=[(X_val,y_val)]
lgbm_wrapper.fit(X_train, y_train, early_stopping_rounds=100, eval_metric="logloss", 
                 eval_set=evals, verbose=True)
preds = lgbm_wrapper.predict(X_test)
pred_proba = lgbm_wrapper.predict_proba(X_test)[:, 1]
get_clf_eval(y_test, preds, pred_proba)


# In[15]:


lgbm_wrapper = LGBMClassifier(n_estimators=1600,max_depth=128,min_child_samples=60,num_leaves=64,subsample=0.8)

# LightGBM도 XGBoost와 동일하게 조기 중단 수행 가능. 
evals=[(X_val,y_val)]
lgbm_wrapper.fit(X_train, y_train, early_stopping_rounds=100, eval_metric="logloss", 
                 eval_set=evals, verbose=True)
preds = lgbm_wrapper.predict(X_test)
pred_proba = lgbm_wrapper.predict_proba(X_test)[:, 1]
get_clf_eval(y_test, preds, pred_proba)


# In[16]:


lgbm_wrapper = LGBMClassifier(n_estimators=3200,max_depth=128,min_child_samples=60,num_leaves=64,subsample=0.8)

# LightGBM도 XGBoost와 동일하게 조기 중단 수행 가능. 
evals=[(X_val,y_val)]
lgbm_wrapper.fit(X_train, y_train, early_stopping_rounds=100, eval_metric="logloss", 
                 eval_set=evals, verbose=True)
preds = lgbm_wrapper.predict(X_test)
pred_proba = lgbm_wrapper.predict_proba(X_test)[:, 1]
get_clf_eval(y_test, preds, pred_proba)


# In[17]:


lgbm_wrapper = LGBMClassifier(n_estimators=1600,max_depth=256,min_child_samples=60,num_leaves=64,subsample=0.8)

# LightGBM도 XGBoost와 동일하게 조기 중단 수행 가능. 
evals=[(X_val,y_val)]
lgbm_wrapper.fit(X_train, y_train, early_stopping_rounds=100, eval_metric="logloss", 
                 eval_set=evals, verbose=True)
preds = lgbm_wrapper.predict(X_test)
pred_proba = lgbm_wrapper.predict_proba(X_test)[:, 1]
get_clf_eval(y_test, preds, pred_proba)


# In[18]:


lgbm_wrapper = LGBMClassifier(n_estimators=1600,max_depth=128,min_child_samples=60,num_leaves=64,subsample=0.8)

# LightGBM도 XGBoost와 동일하게 조기 중단 수행 가능. 
evals=[(X_val,y_val)]
lgbm_wrapper.fit(X_train, y_train, early_stopping_rounds=100, eval_metric="logloss", 
                 eval_set=evals, verbose=True)
preds = lgbm_wrapper.predict(X_test)
pred_proba = lgbm_wrapper.predict_proba(X_test)[:, 1]
get_clf_eval(y_test, preds, pred_proba)


# In[19]:


lgbm_wrapper = LGBMClassifier(n_estimators=1600,max_depth=128,min_child_samples=60,num_leaves=64,subsample=0.8)

# LightGBM도 XGBoost와 동일하게 조기 중단 수행 가능. 
evals=[(X_val,y_val)]
lgbm_wrapper.fit(X_train, y_train, early_stopping_rounds=100, eval_metric="logloss", 
                 eval_set=evals, verbose=True)
preds = lgbm_wrapper.predict(X_test)
pred_proba = lgbm_wrapper.predict_proba(X_test)[:, 1]
get_clf_eval(y_test, preds, pred_proba)


# In[23]:


lgbm_wrapper = LGBMClassifier(n_estimators=1400,max_depth=128,min_child_samples=80,num_leaves=81,subsample=0.8)

# LightGBM도 XGBoost와 동일하게 조기 중단 수행 가능. 
evals=[(X_val,y_val)]
lgbm_wrapper.fit(X_train, y_train, early_stopping_rounds=100, eval_metric="logloss", 
                 eval_set=evals, verbose=True)
preds = lgbm_wrapper.predict(X_test)
pred_proba = lgbm_wrapper.predict_proba(X_test)[:, 1]
get_clf_eval(y_test, preds, pred_proba)


# In[24]:


lgbm_wrapper = LGBMClassifier(n_estimators=1600,max_depth=128,min_child_samples=80,num_leaves=64,subsample=0.8)

# LightGBM도 XGBoost와 동일하게 조기 중단 수행 가능. 
evals=[(X_val,y_val)]
lgbm_wrapper.fit(X_train, y_train, early_stopping_rounds=100, eval_metric="logloss", 
                 eval_set=evals, verbose=True)
preds = lgbm_wrapper.predict(X_test)
pred_proba = lgbm_wrapper.predict_proba(X_test)[:, 1]
get_clf_eval(y_test, preds, pred_proba)


# In[25]:


lgbm_wrapper = LGBMClassifier(n_estimators=1600,max_depth=128,min_child_samples=60,num_leaves=80,subsample=0.8)

# LightGBM도 XGBoost와 동일하게 조기 중단 수행 가능. 
evals=[(X_val,y_val)]
lgbm_wrapper.fit(X_train, y_train, early_stopping_rounds=100, eval_metric="logloss", 
                 eval_set=evals, verbose=True)
preds = lgbm_wrapper.predict(X_test)
pred_proba = lgbm_wrapper.predict_proba(X_test)[:, 1]
get_clf_eval(y_test, preds, pred_proba)


# In[26]:


lgbm_wrapper = LGBMClassifier(n_estimators=1600,max_depth=128,min_child_samples=60,num_leaves=100,subsample=0.8)

# LightGBM도 XGBoost와 동일하게 조기 중단 수행 가능. 
evals=[(X_val,y_val)]
lgbm_wrapper.fit(X_train, y_train, early_stopping_rounds=100, eval_metric="logloss", 
                 eval_set=evals, verbose=True)
preds = lgbm_wrapper.predict(X_test)
pred_proba = lgbm_wrapper.predict_proba(X_test)[:, 1]
get_clf_eval(y_test, preds, pred_proba)


# In[27]:


lgbm_wrapper = LGBMClassifier(n_estimators=1600,max_depth=128,min_child_samples=60,num_leaves=100,subsample=0.7)

# LightGBM도 XGBoost와 동일하게 조기 중단 수행 가능. 
evals=[(X_val,y_val)]
lgbm_wrapper.fit(X_train, y_train, early_stopping_rounds=100, eval_metric="logloss", 
                 eval_set=evals, verbose=True)
preds = lgbm_wrapper.predict(X_test)
pred_proba = lgbm_wrapper.predict_proba(X_test)[:, 1]
get_clf_eval(y_test, preds, pred_proba)


# In[10]:


lgbm_wrapper = LGBMClassifier(n_estimators=1600,max_depth=128,min_child_samples=60,num_leaves=80,subsample=0.7)

# LightGBM도 XGBoost와 동일하게 조기 중단 수행 가능. 
evals=[(X_val,y_val)]
lgbm_wrapper.fit(X_train, y_train, early_stopping_rounds=100, eval_metric="logloss", 
                 eval_set=evals, verbose=True)
preds = lgbm_wrapper.predict(X_test)
pred_proba = lgbm_wrapper.predict_proba(X_test)[:, 1]
get_clf_eval(y_test, preds, pred_proba)


# In[11]:


lgbm_wrapper = LGBMClassifier(n_estimators=1600,max_depth=128,min_child_samples=60,num_leaves=80,subsample=0.6)

# LightGBM도 XGBoost와 동일하게 조기 중단 수행 가능. 
evals=[(X_val,y_val)]
lgbm_wrapper.fit(X_train, y_train, early_stopping_rounds=100, eval_metric="logloss", 
                 eval_set=evals, verbose=True)
preds = lgbm_wrapper.predict(X_test)
pred_proba = lgbm_wrapper.predict_proba(X_test)[:, 1]
get_clf_eval(y_test, preds, pred_proba)


# In[ ]:




