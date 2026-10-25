from pathlib import Path
import json, joblib
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE

ROOT=Path(__file__).resolve().parents[1]
df=pd.read_csv(ROOT/"data"/"fraud_transactions_demo.csv")
df["Amount_Log"]=np.log1p(df["Amount"])
df["Night_Transaction"]=((df["Hour"]<=5)|(df["Hour"]>=23)).astype(int)

X=df.drop(columns=["Transaction_ID","Class"])
y=df["Class"]
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)

imp=SimpleImputer(strategy="median")
Xtr=imp.fit_transform(Xtr)
Xte=imp.transform(Xte)

sc=StandardScaler()
Xtr=sc.fit_transform(Xtr)
Xte=sc.transform(Xte)

Xtr,ytr=SMOTE(random_state=42).fit_resample(Xtr,ytr)

models={
"Logistic Regression":LogisticRegression(max_iter=1500,class_weight="balanced",random_state=42),
"Decision Tree":DecisionTreeClassifier(max_depth=8,class_weight="balanced",random_state=42),
"Random Forest":RandomForestClassifier(n_estimators=200,max_depth=12,class_weight="balanced",random_state=42,n_jobs=-1)
}

results=[]
preds={}
for name,model in models.items():
    model.fit(Xtr,ytr)
    pred=model.predict(Xte)
    prob=model.predict_proba(Xte)[:,1]
    preds[name]=pred
    results.append([name,precision_score(yte,pred,zero_division=0),recall_score(yte,pred,zero_division=0),f1_score(yte,pred,zero_division=0),roc_auc_score(yte,prob)])

res=pd.DataFrame(results,columns=["Model","Precision","Recall","F1_Score","ROC_AUC"]).sort_values("F1_Score",ascending=False)
best=str(res.iloc[0]["Model"])

for d in ["models","reports","figures","screenshots"]:
    (ROOT/d).mkdir(exist_ok=True)

res.to_csv(ROOT/"reports"/"model_comparison.csv",index=False)
joblib.dump(models[best],ROOT/"models"/"best_fraud_model.joblib")
joblib.dump(imp,ROOT/"models"/"imputer.joblib")
joblib.dump(sc,ROOT/"models"/"scaler.joblib")
(ROOT/"models"/"model_metadata.json").write_text(json.dumps({"best_model":best,"features":list(X.columns)},indent=2))

np.save(ROOT/"reports"/"test_labels.npy",yte.to_numpy())
np.save(ROOT/"reports"/"best_predictions.npy",preds[best])

print(res.to_string(index=False))
print("BEST:",best)
