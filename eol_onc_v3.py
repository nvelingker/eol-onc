import pandas as pd
labs_df = pd.read_csv('/workspace/data/labs.csv')
stage_df = pd.read_csv('/workspace/data/stage.csv')
dx_df = pd.read_csv('/workspace/data/dx.csv')
pat_list_df = pd.read_csv('/workspace/data/pat_list.csv')
def merge_on_pat_id_and_empi(main_df, merge_df, date_col, appt_col='APPT_TIME'):
    if 'EMPI' in merge_df.columns:
        merged_df = pd.merge(main_df[['PAT_ID', 'EMPI', appt_col]], merge_df, on=['PAT_ID', 'EMPI'], how='left')
    else:
        merged_df = pd.merge(main_df[['PAT_ID', 'EMPI', appt_col]], merge_df, on='PAT_ID', how='left')
    merged_df[date_col] = pd.to_datetime(merged_df[date_col]).dt.date
    merged_df[appt_col] = pd.to_datetime(merged_df[appt_col]).dt.date
    filtered_df = merged_df[merged_df[date_col] < merged_df[appt_col]]
    return filtered_df
dx_merged_filtered = merge_on_pat_id_and_empi(pat_list_df, dx_df, 'CONTACT_DATE')
labs_merged_filtered = merge_on_pat_id_and_empi(pat_list_df, labs_df, 'RESULT_TIME')
stage_merged_filtered = merge_on_pat_id_and_empi(pat_list_df, stage_df, 'SIGN_DATETIME')
dx_grouped_efficient = dx_merged_filtered.groupby(['PAT_ID', 'EMPI', 'APPT_TIME']).agg({'CODE': lambda x: list(x)}).reset_index()
dx_grouped_efficient.rename(columns={'CODE': 'AGGREGATED_CODES'}, inplace=True)
stage_indicators = pd.get_dummies(stage_merged_filtered, columns=['BODY_SITE_NAME', 'STAGE_GROUP'])
stage_grouped = stage_indicators.groupby(['PAT_ID', 'EMPI', 'APPT_TIME']).sum().reset_index()
original_columns = ['PAT_ID', 'EMPI', 'APPT_TIME']
aggregated_columns = [col for col in stage_grouped.columns if col in original_columns or col.startswith(('BODY_SITE_NAME_', 'STAGE_GROUP_'))]
stage_grouped_final = stage_grouped[aggregated_columns]
stage_grouped_final.to_csv("/workspace/stage_feat.csv")
lab_tests_list = [
    "HEMATOCRIT",
    "HEMOGLOBIN",
    "WHITE BLOOD COUNT",
    "MCV",
    "PLATELET COUNT",
    "MCHC",
    "RED BLOOD CELL COUNT",
    "MCH",
    "POTASSIUM",
    "SODIUM",
    "CREATININE",
    "BUN",
    "GLUCOSE",
    "EGFR CKD-EPI CR 2021",
    "CHLORIDE",
    "CALCIUM",
    "CO2",
    "RDW",
    "TOTAL PROTEIN",
    "ALBUMIN",
    "LYMPHOCYTE",
    "NEUTROPHIL",
    "MONOCYTE",
    "EOSINOPHIL",
    "BASOPHIL",
    "ABSOLUTE MONOCYTES",
    "ABSOLUTE LYMPHOCYTES",
    "ABS NEUTROPHILS",
    "ABSOLUTE BASOPHILS",
    "ABSOLUTE EOSINOPHILS"
]
labs_filtered_by_common_name = labs_merged_filtered[labs_merged_filtered['COMMON_NAME'].isin(lab_tests_list)]
aggregated_features = pd.DataFrame()
for test in lab_tests_list:
    test_df = labs_filtered_by_common_name[labs_filtered_by_common_name['COMMON_NAME'] == test]
    most_recent_value = test_df.groupby(['PAT_ID', 'EMPI', 'APPT_TIME']).agg({
        'ORD_NUM_VALUE': 'last',
        'RESULT_FLAG': 'last'
    }).rename(columns={'ORD_NUM_VALUE': f"{test}_MOST_RECENT_VALUE", 'RESULT_FLAG': f"{test}_MOST_RECENT_FLAG"})
    test_count = test_df.groupby(['PAT_ID', 'EMPI', 'APPT_TIME']).size().to_frame(name=f"{test}_COUNT")
    if aggregated_features.empty:
        aggregated_features = most_recent_value.join(test_count, how='outer')
    else:
        aggregated_features = aggregated_features.join(most_recent_value, how='outer').join(test_count, how='outer')
aggregated_features.reset_index(inplace=True)
aggregated_features.to_csv("/workspace/lab_feat.csv")
from hcuppy.elixhauser import ElixhauserEngine
import pandas as pd
ee = ElixhauserEngine()
def apply_elixhauser(row):
    dx_full_lst = row['AGGREGATED_CODES']
    out = ee.get_elixhauser(dx_full_lst)
    features = {}
    for cmrbdt in out['cmrbdt_lst']:
        features[f"cmrbdt_{cmrbdt}"] = 1
    features['mrtlt_scr'] = out['mrtlt_scr']
    features['rdmsn_scr'] = out['rdmsn_scr']
    return pd.Series(features)
feature_set = dx_grouped_efficient.apply(apply_elixhauser, axis=1)
dx_grouped_with_features = pd.concat([dx_grouped_efficient, feature_set], axis=1)
dx_grouped_with_features.fillna(0, inplace=True)
for column in dx_grouped_with_features.columns:
    if column.startswith('cmrbdt_'):
        dx_grouped_with_features[column] = dx_grouped_with_features[column].astype(int)
dx_grouped_with_features = dx_grouped_with_features.drop("AGGREGATED_CODES", axis = 1)
dx_grouped_with_features.to_csv("/workspace/dx_feat.csv")
pat_list_df['APPT_TIME'] = pd.to_datetime(pat_list_df['APPT_TIME']).dt.date
pat_list_df['BIRTH_DATE'] = pd.to_datetime(pat_list_df['BIRTH_DATE']).dt.date
pat_list_df['DEATH_DATE'] = pd.to_datetime(pat_list_df['DEATH_DATE'], errors='coerce').dt.date  
pat_list_df['SEX_INDICATOR'] = (pat_list_df['SEX_C'] - 1).astype(int)  
pat_list_df['AGE'] = (pat_list_df['APPT_TIME'] - pat_list_df['BIRTH_DATE']).dt.days // 365
grouped_features = pat_list_df.groupby(['PAT_ID', 'EMPI', 'APPT_TIME']).agg({
    'SEX_INDICATOR': 'first',  
    'AGE': 'first',  
}).reset_index()
pat_list_df['DEAD'] = pat_list_df['DEATH_DATE'].notnull().astype(int)
mortality_flag = pat_list_df.groupby(['PAT_ID', 'EMPI'])['DEAD'].max().reset_index()  
pat_id_feats = pd.merge(grouped_features, mortality_flag, on=['PAT_ID', 'EMPI'], how='left')
pat_id_feats.to_csv("/workspace/pat_id_feat.csv")
lab_feat_df = pd.read_csv('/workspace/lab_feat.csv')
stage_feat_df = pd.read_csv('/workspace/stage_feat.csv')
dx_feat_df = pd.read_csv('/workspace/dx_feat.csv')
pat_id_feat_df = pd.read_csv('/workspace/pat_id_feat.csv')
base_df = pat_id_feat_df
merged_df = pd.merge(base_df, dx_feat_df, on=["PAT_ID", "EMPI", "APPT_TIME"], how='left', suffixes=('', '_dx'))
merged_df = pd.merge(merged_df, lab_feat_df, on=["PAT_ID", "EMPI", "APPT_TIME"], how='left', suffixes=('', '_pat'))
merged_df = pd.merge(merged_df, stage_feat_df, on=["PAT_ID", "EMPI", "APPT_TIME"], how='left', suffixes=('', '_stage'))
merged_df.fillna(0, inplace=True)
cleaned_df = merged_df.fillna(0)
def drop_unwanted_columns(df, columns_to_exclude=['PAT_ID', 'EMPI', 'APPT_TIME']):  
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')] 
    df = df.drop(columns=[col for col in columns_to_exclude if col in df.columns], errors='ignore')
    return df
import pandas as pd
def convert_string_columns_to_indicators(df):
    
    string_cols = df.select_dtypes(include=['object']).columns
    
    df_with_dummies = pd.get_dummies(df, columns=string_cols, drop_first=True)
    return df_with_dummies
cleaned_df = drop_unwanted_columns(cleaned_df)
d = {'Low Panic': 0.25, 'Low':0.5,'High':0.75,'High Panic':1}
def replace_flag_values(df, replacement_dict):
    
    for column in df.columns:
        
        if column.endswith('FLAG'):
            
            df[column] = df[column].replace(replacement_dict)
    return df
cleaned_df = replace_flag_values(cleaned_df, d)
list(cleaned_df.columns)
cleaned_df.to_csv("/workspace/cleaned_features.csv")
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import accuracy_score
def train_best_random_forest(X, y):
    param_grid = {
        'n_estimators': [100, 200],  
        'max_depth': [None, 10, 20],  
        'min_samples_split': [2, 5],  
        'min_samples_leaf': [1, 2],  
    }
    rf = RandomForestClassifier()
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv, scoring='accuracy', verbose=1, n_jobs=-1)
    
    grid_search.fit(X, y)
    
    best_model = grid_search.best_estimator_
    best_score = grid_search.best_score_
    return best_model, best_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix
def train_best_random_forest(X, y):
    
    param_grid = {
        'n_estimators': [100, 200, 300],  
        'max_depth': [10, 20, 30, None],  
        'min_samples_split': [2, 5, 10],  
        'min_samples_leaf': [1, 2, 4],  
        'max_features': ['sqrt', 'log2', None],  
        'bootstrap': [True, False]  
    }
    
    rf = RandomForestClassifier()
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='accuracy', verbose=1, n_jobs=-1)
    
    grid_search.fit(X, y)
    
    best_model = grid_search.best_estimator_
    best_score = grid_search.best_score_
    return best_model, best_score
X = cleaned_df.drop('DEAD', axis=1)
y = cleaned_df['DEAD']
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
best_rf_model, best_rf_score = train_best_random_forest(X_train, y_train)
y_train_probs = best_rf_model.predict_proba(X_train)[:, 1]
y_test_probs = best_rf_model.predict_proba(X_test)[:, 1]
train_auc = roc_auc_score(y_train, y_train_probs)
test_auc = roc_auc_score(y_test, y_test_probs)
train_auprc = average_precision_score(y_train, y_train_probs)
test_auprc = average_precision_score(y_test, y_test_probs)
def calculate_tpr_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return tpr, specificity
y_train_pred = best_rf_model.predict(X_train)
y_test_pred = best_rf_model.predict(X_test)
train_tpr, train_specificity = calculate_tpr_specificity(y_train, y_train_pred)
test_tpr, test_specificity = calculate_tpr_specificity(y_test, y_test_pred)
print("Training AUC: {:.4f}, Training AUPRC: {:.4f}, Training TPR: {:.4f}, Training Specificity: {:.4f}".format(train_auc, train_auprc, train_tpr, train_specificity))
print("Test AUC: {:.4f}, Test AUPRC: {:.4f}, Test TPR: {:.4f}, Test Specificity: {:.4f}".format(test_auc, test_auprc, test_tpr, test_specificity))
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, precision_recall_curve
def calculate_metrics_at_thresholds(y_true, y_probs):
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    metrics = []
    for threshold in thresholds:
        
        y_pred = (y_probs >= threshold).astype(int)
        
        auc_score = roc_auc_score(y_true, y_probs)  
        auprc_score = average_precision_score(y_true, y_probs)  
        tpr, specificity = calculate_tpr_specificity(y_true, y_pred)   
        metrics.append({
            'threshold': threshold,
            'AUC': auc_score,
            'AUPRC': auprc_score,
            'TPR': tpr,
            'Specificity': specificity,
            'confusion': str(confusion_matrix(y_true, y_pred))
        })
    return metrics
def calculate_tpr_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return tpr, specificity
train_metrics = calculate_metrics_at_thresholds(y_train, y_train_probs)
test_metrics = calculate_metrics_at_thresholds(y_test, y_test_probs)
for metric in train_metrics:
    print(f"Threshold: {metric['threshold']}, Training AUC: {metric['AUC']:.4f}, Training AUPRC: {metric['AUPRC']:.4f}, Training TPR: {metric['TPR']:.4f}, Training Specificity: {metric['Specificity']:.4f}, \nConfusion:\n{metric['confusion']}")
for metric in test_metrics:
    print(f"Threshold: {metric['threshold']}, Test AUC: {metric['AUC']:.4f}, Test AUPRC: {metric['AUPRC']:.4f}, Test TPR: {metric['TPR']:.4f}, Test Specificity: {metric['Specificity']:.4f}, \nConfusion: \n{metric['confusion']}")
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix
def train_best_gradient_boosting(X, y):
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],  
        'n_estimators': [100, 200, 300],  
        'subsample': [0.8, 0.9, 1.0],  
        'max_depth': [3, 5, 7],  
        'min_samples_split': [2, 4],  
        'min_samples_leaf': [1, 2],  
        'max_features': ['sqrt', 'log2', None]  
    }
    gb = GradientBoostingClassifier()
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(estimator=gb, param_grid=param_grid, scoring='accuracy', verbose=1, n_jobs=-1)
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_
    best_score = grid_search.best_score_
    return best_model, best_score
best_gb_model, best_gb_score = train_best_gradient_boosting(X_train, y_train)
y_train_probs_gb = best_gb_model.predict_proba(X_train)[:, 1]
y_test_probs_gb = best_gb_model.predict_proba(X_test)[:, 1]
train_auc_gb = roc_auc_score(y_train, y_train_probs_gb)
test_auc_gb = roc_auc_score(y_test, y_test_probs_gb)
train_auprc_gb = average_precision_score(y_train, y_train_probs_gb)
test_auprc_gb = average_precision_score(y_test, y_test_probs_gb)
y_train_pred_gb = best_gb_model.predict(X_train)
y_test_pred_gb = best_gb_model.predict(X_test)
train_tpr_gb, train_specificity_gb = calculate_tpr_specificity(y_train, y_train_pred_gb)
test_tpr_gb, test_specificity_gb = calculate_tpr_specificity(y_test, y_test_pred_gb)
print("Gradient Boosting Training AUC: {:.4f}, Training AUPRC: {:.4f}, Training TPR: {:.4f}, Training Specificity: {:.4f}".format(train_auc_gb, train_auprc_gb, train_tpr_gb, train_specificity_gb))
print("Gradient Boosting Test AUC: {:.4f}, Test AUPRC: {:.4f}, Test TPR: {:.4f}, Test Specificity: {:.4f}".format(test_auc_gb, test_auprc_gb, test_tpr_gb, test_specificity_gb))
train_metrics_gb = calculate_metrics_at_thresholds(y_train, y_train_probs_gb)
test_metrics_gb = calculate_metrics_at_thresholds(y_test, y_test_probs_gb)
for metric in train_metrics_gb:
    print(f"Threshold: {metric['threshold']}, Training AUC: {metric['AUC']:.4f}, Training AUPRC: {metric['AUPRC']:.4f}, Training TPR: {metric['TPR']:.4f}, Training Specificity: {metric['Specificity']:.4f}, \nConfusion:\n{metric['confusion']}")
for metric in test_metrics_gb:
    print(f"Threshold: {metric['threshold']}, Test AUC: {metric['AUC']:.4f}, Test AUPRC: {metric['AUPRC']:.4f}, Test TPR: {metric['TPR']:.4f}, Test Specificity: {metric['Specificity']:.4f}, \nConfusion: \n{metric['confusion']}")
