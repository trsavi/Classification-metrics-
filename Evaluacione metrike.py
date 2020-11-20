#!/usr/bin/env python
# coding: utf-8

# # Definisanje metoda

# Ucitavanje biblioteka i dataseta 

# In[257]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
df = pd.read_csv("data.csv")


# In[258]:


df.head()


# Posto nam trebaju binarne vrednosti, odredjujemo vrednosti u koloni na osnovu threshold-a

# In[259]:


def threshold(thresh):
    df['model_LR_bin'] = df['model_LR'].apply(lambda x: 1 if x>thresh else 0)
    df['model_RF_bin'] = df['model_RF'].apply(lambda x: 1 if x>thresh else 0)


# In[260]:


threshold(0.5)


# In[261]:


df.head()


# Definisanje True Positive, True Negative, False Positive i False Negative vrednosti u zasebne kolone

#  # Evaluacione metrike
#  
#  Napomena: Nisu radjene srednje vrednosti za sve klase vec je za svaku klasu posebno izracunata metrika sem kod accuracy

# ## Matrica konfuzije koja podrzava i multiclass probleme

# In[262]:


def confusion_matrix(y_true, y_pred):
    classes=np.unique(y_true)
    dim=len(classes)
    cm=np.zeros((dim,dim))
    for i in range(len(y_true)):
        cm[y_true[i], y_pred[i]]+=1
    return cm.astype(np.int)


# In[263]:


y_true = df['actual_label']
y_pred = df['model_RF_bin']
cm = confusion_matrix(y_true,y_pred)
cm


# ## Accuracy
# 
# Izracunava se kao: (TP+TN)/total

# In[264]:


def accuracy(matrix):
    return (sum(np.diagonal(matrix)))/np.sum(matrix)


# In[265]:


print("Accuracy for Linear Model")
accuracy(matrix_L)


# In[266]:


print("Accuracy for RF Model")
accuracy(matrix_R)


# ## Precision
# 
# Izracunava se kao: TP/(TP+FP) 

# In[267]:


def precision(matrix):
    for i in range(len(matrix)):
        print("For class {}, precision is: {}".format(i,((matrix[i][i])/sum(matrix[i]))))


# In[268]:


print("Precision for Linear Model")
precision(matrix_L)


# In[269]:


print("Precision for RF Model")
precision(matrix_R)


# ## Recall/Sensitivity
# 
# Izracunava se kao: TP/(TP+FN)

# In[270]:


def recall(matrix, roc=False):
    if roc:
        lista=[]
        for i in range(len(matrix)):
            lista.append(((matrix[i][i])/sum(matrix[:,i])) if sum(matrix[:,i]) > 0 else 0)
        return lista
    else:
        for i in range(len(matrix)):
            print("For class {}, precision is: {}".format(i,((matrix[i][i])/sum(matrix[:][i]))))


# In[271]:


print("Recall for Linear Model")
recall(matrix_L)


# In[272]:


print("Recall for RF Model")
recall(matrix_R)


# ## Specificity
# 
# Izracunava se kao: TN/(TN+FP)

# In[273]:


def specificity(matrix):
    num_classes = np.shape(matrix)[0]
    for j in range(num_classes):
        
        tp = np.sum(matrix[j][j])
        fp = np.sum(matrix[j][int(np.concatenate((np.arange(0, j), np.arange(j+1, num_classes))))])
        fn = np.sum(matrix[int(np.concatenate((np.arange(0, j), np.arange(j+1, num_classes))))][j])
        tn = np.sum(matrix) - (tp+fp+fn)
        print("For class {}, recall is: {}".format(j,(tn/(tn+fp))))


# In[274]:


print("Specificity for Linear Model")
specificity(matrix_L)


# In[275]:


print("Specificity for RF Model")
specificity(matrix_R)


# ## F-1 score
# 
# Izracunava se kao: 2*TP/(2*TP+FN+FP)

# In[276]:


# Podrzava i multiclass probleme
def f1(matrix):
    num_classes = np.shape(matrix)[0]
    f1_score = np.zeros(shape=(num_classes,), dtype='float32')
    weights = np.sum(matrix, axis=0)/np.sum(matrix)

    for j in range(num_classes):
        tp = np.sum(matrix[j][j])
        fp = np.sum(matrix[j][int(np.concatenate((np.arange(0, j), np.arange(j+1, num_classes))))])
        fn = np.sum(matrix[int(np.concatenate((np.arange(0, j), np.arange(j+1, num_classes))))][j])
        precision = tp/(tp+fp) if (tp+fp) > 0 else 0
        recall = tp/(tp+fn) if (tp+fn) > 0 else 0
        f1_score[j] = 2*precision*recall/(precision + recall)*weights[j] if (precision + recall) > 0 else 0

    f1_score = np.sum(f1_score)
    return f1_score


# In[277]:


print("F1-score for Linear Model")
f1(matrix_L)


# In[278]:


print("F1-score for RF Model")
f1(matrix_R)


# In[ ]:




