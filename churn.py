import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

## ML
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, plot_roc_curve## Algorithms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer

pd.set_option('display.max_columns', None)

raw_data = pd.read_csv("churn.csv")

print(raw_data.shape)
print(raw_data.info())
print(raw_data.describe())
print(raw_data.head())



#let's check for duplicated data
duplicates = raw_data.duplicated()
print("printing duplicates")
print(duplicates.sum())
#no duplicates so we don't have to drop them

#drop customer id as it has no valuable info
raw_data = raw_data.drop("customerID", axis=1)

#some data in total charge column has " " as a value
incorrect_charges = (raw_data['TotalCharges'] == ' ')
print("incorrect charges")
print(incorrect_charges.sum())
#lets find what fraction of data has this wrong values
print(incorrect_charges.sum()/raw_data.shape[0]*100)
#as it is around 0.16% of our data instead of imputing missing data it is fine to drop it
raw_data['TotalCharges'] = raw_data['TotalCharges'].replace(' ', np.nan)
raw_data.dropna(subset=['TotalCharges'], inplace=True)
raw_data["TotalCharges"] = raw_data["TotalCharges"].astype("float")


#plot histograms to explore data distribution
fig, ax = plt.subplots(4, 5, figsize=(22, 20))

plt.subplots_adjust(left=None, bottom=None, right=None, top=0.9, wspace=0.5, hspace=0.6)
for column, subplot in zip(raw_data.columns, ax.flatten()):
    hist = sns.histplot(raw_data[column], ax=subplot)
    hist.tick_params(axis='x', rotation=15, labelsize=8)
plt.show()

#plot pairplots for bivariate analysis
sns.pairplot(data = raw_data, hue='Churn')
plt.show()


#explore how gender, seniority, partner and dependents impact churn
columns_to_plot = ["gender","SeniorCitizen","Partner","Dependents"]
fig, ax = plt.subplots(2, 2, figsize=(22, 20))
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.9, wspace=0.5, hspace=0.6)
for column, subplot in zip(columns_to_plot, ax.flatten()):
    sns.countplot(x=column, hue='Churn', data=raw_data, ax=subplot)
plt.show()


#convert churn to 0 or 1
raw_data['Churn'] = raw_data['Churn'].map( {'No': 0, 'Yes': 1} ).astype(int)

#create heatmaps to explore how multiple variables affect churn
def create_pivot(data,index,columns,value="Churn"):
    result = pd.pivot_table(data=data, index=index, columns=columns,values=value)
    return result



heatmaps = [["PaymentMethod","Contract"],["Partner","Dependents"],["PhoneService","InternetService"],["gender","SeniorCitizen"]]
fig, ax = plt.subplots(2, 2, figsize=(22, 20))
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.9, wspace=0.5, hspace=0.6)
for heatmap, subplot in zip(heatmaps, ax.flatten()):
    pivot = create_pivot(raw_data,heatmap[0],heatmap[1])
    sns.heatmap(pivot, annot=True, cmap='RdYlGn_r',ax=subplot).set_title('How does {var1} and {var2} affect churn?'.format(var1=heatmap[0],var2=heatmap[1]))
plt.show()

#train test split, leave 30% of dataset for evaluation
x = raw_data.drop(["Churn"], axis=1)
y = raw_data["Churn"]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

numerical_ix = X_train.select_dtypes(include=np.number).columns
categorical_ix = X_train.select_dtypes(exclude=np.number).columns

#create preprocessing pipelines for each datatype
numerical_transformer = Pipeline(steps=[
('imputer', SimpleImputer(strategy='median')),
('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
('encoder', OrdinalEncoder()),
('scaler', StandardScaler())])

#put the preprocessing steps together
preprocessor = ColumnTransformer([
('numerical', numerical_transformer, numerical_ix),
('categorical', categorical_transformer, categorical_ix)],
remainder='passthrough')

#let's try scikit classification models
classifiers = [
KNeighborsClassifier(),
SVC(random_state=42),
DecisionTreeClassifier(random_state=42),
RandomForestClassifier(random_state=42),
AdaBoostClassifier(random_state=42),
GradientBoostingClassifier(random_state=42)
]
classifier_names = [
'KNeighborsClassifier()',
'SVC()',
'DecisionTreeClassifier()',
'RandomForestClassifier()',
'AdaBoostClassifier()',
'GradientBoostingClassifier()'
]
model_scores = []
#loop through the classifiers
#using roc_auc as it is the best metrics for binary classification when the dataset is imbalanced
selector = SelectKBest(k=len(X_train.columns))
for classifier, name in zip(classifiers, classifier_names):
  pipe = Pipeline(steps=[
  ('preprocessor', preprocessor),
  ('selector', selector),
  ('classifier', classifier)])
  score = cross_val_score(pipe, X_train, y_train, cv=10, scoring='roc_auc').mean()
  model_scores.append(score)


model_performance = pd.DataFrame({
  'Classifier':
    classifier_names,
  'Cross-validated AUC':
    model_scores
}).sort_values('Cross-validated AUC', ascending = False, ignore_index=True)
print(model_performance)

#gradient boosting classifier scores the best, so lets fine tune its hyperparameters

pipe = Pipeline(steps=[
  ('preprocessor', preprocessor),
  ('selector', selector),
  ('classifier', GradientBoostingClassifier(random_state=42))
])

grid = {
  "classifier__max_depth":[1,2,3,5,10],
  "classifier__learning_rate":[0.01,0.1,0.5,1],
  "classifier__n_estimators":[100,200,300,400,500]
}
gridsearch = GridSearchCV(estimator=pipe, param_grid=grid, n_jobs= 5, scoring='roc_auc')
gridsearch.fit(X_train, y_train)
print(gridsearch.best_params_)
print(gridsearch.best_score_)
