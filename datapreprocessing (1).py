import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stat
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('incident_event_log.csv')
print(df.info())

missing_percentages = []
total_length = len(df) 

col_l = []
def percent(f,col,total_length):
    per = (len(f)/total_length) * 100
    print(f"Percentage of missing values for {col}: {per:.2f}")
    if per >= 40:
        col_l.append(col)

for col in df.columns:
    d_miss = df[df[col] == '?'][col]
    percent(d_miss,col,total_length)

missing_percentages = []
for col in df.columns:
    missing_count = len(df[df[col] == '?'][col])
    missing_percentage = (missing_count / len(df)) * 100
    missing_percentages.append({'Column': col, 'Percentage Missing': missing_percentage})

# Create a DataFrame from the list of dictionaries
missing_values_df = pd.DataFrame(missing_percentages)

# Filter columns with missing percentage >= 40
col_l = missing_values_df[missing_values_df['Percentage Missing'] >= 40]

# Plot the bar chart
plt.figure(figsize=(10, 6))
plt.bar(missing_values_df['Column'], missing_values_df['Percentage Missing'], color='skyblue')

# Add labels and title
plt.xlabel('Column')
plt.ylabel('Percentage of Missing Values (%)')
plt.title('Percentage of Missing Values in Columns')

# Rotate x-axis labels for better readability
plt.xticks(rotation=90)

# Show the plot
plt.show()

for col in df.columns:
    missing_count = len(df[df[col] == '?'][col])
    missing_percentage = (missing_count / total_length) * 100
    missing_percentages.append({'Column': col, 'Percentage Missing': missing_percentage})

# Create a DataFrame from the list of dictionaries
missing_values_df = pd.DataFrame(missing_percentages)

# Filter columns with missing percentage >= 40
col_l = missing_values_df[missing_values_df['Percentage Missing'] >= 40]

# Plot the bar chart
plt.figure(figsize=(10, 6))
plt.bar(missing_values_df['Column'], missing_values_df['Percentage Missing'], color='skyblue')

# Add labels and title
plt.xlabel('Column')
plt.ylabel('Percentage of Missing Values (%)')
plt.title('Percentage of Missing Values in Columns')

# Rotate x-axis labels for better readability
plt.xticks(rotation=90)

# Show the plot
plt.show()

# Display the DataFrame
print(col_l)

print(f"\nDropping columns: {col_l}")
# Create a list of column names to drop (those with missing data percentage >= 40)
columns_to_drop = col_l['Column'].tolist()

print(f"\nDropping columns: {columns_to_drop}")
df.drop(columns=columns_to_drop, inplace=True)


columns_to_check = ['caller_id', 'opened_by', 'location', 'category', 'subcategory', 'closed_code', 'resolved_by', 'resolved_at']
in_tbr = list(set(df[df[columns_to_check].isin(['?']).any(axis=1)]['number']))
df_cleaned = df[~df['number'].isin(in_tbr)]

columns_to_check = ['caller_id','opened_by','sys_created_by','sys_updated_by','location','category','subcategory','u_symptom','impact','urgency','priority','assignment_group','assigned_to','closed_code','resolved_by']

def extract_digits(text):
    digits = re.findall(r'\d+', str(text))  # Find all sequences of digits
    if digits:
        return int(''.join(digits))  # Join the digits and convert to integer
    else:
        return None  # Return None if no digits found


for col in columns_to_check:
    fname = f"{col}_num"
    df_cleaned[fname] = df_cleaned[col].apply(extract_digits)

df_cleaned =df_cleaned.reset_index(drop=True)
label_enc = ['incident_state','active','made_sla','contact_type','knowledge','u_priority_confirmation','notify']

for col in label_enc:
    fname = f"{col}_encoded"
    label_encoder = LabelEncoder()
    df_cleaned[fname] = label_encoder.fit_transform(df_cleaned[col])

numeric_df = df_cleaned.select_dtypes(include=['number']).reset_index(drop=True)

#print(numeric_df.info())

def feature_imp(rf_classifier,X_train,Target):
    feature_importances = rf_classifier.feature_importances_

    # Sort feature importances in descending order
    sorted_indices = feature_importances.argsort()[::-1]

    # Plot feature importances
    plt.figure(figsize=(20, 8))
    plt.bar(range(len(feature_importances)), feature_importances[sorted_indices])
    plt.xticks(range(len(feature_importances)), X_train.columns[sorted_indices], rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title(f'Random Forest Feature Importance for: {Target}')
    plt.show()


#X = df_train.drop(['assigned_to_num'], axis=1)
#y = df_train['assigned_to_num']
def rmodel_acc(df_train,df_test,target,flag):
    X = df_train.drop(target,axis=1)
    y = df_train[target]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_missing = df_test
    # Initialize and train a Random Forest model
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    if flag ==1:
        feature_imp(rf_model, X_train,target)
    else:
    # Evaluate model performance
        y_train_pred = rf_model.predict(X_train)
        y_test_pred = rf_model.predict(X_test)

        imputed_values = rf_model.predict(X_missing)

        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        print(f"Train Accuracy: {target}", train_accuracy)
        print(f"Test Accuracy: {target}", test_accuracy)
        return imputed_values

cols_to_fillna = ['assignment_group_num','assigned_to_num','u_symptom_num','sys_created_by_num']
imp_df = numeric_df
for i in range(0,len(cols_to_fillna)):
    if i == 0:
      train_cols = ['resolved_by_num','category_num','closed_code_num','subcategory_num']
      target = cols_to_fillna[i]
      train_cols.append(target)
      #continue
    elif i == 1:
      train_cols = ['resolved_by_num','assignment_group_num','sys_updated_by_num','category_num']
      target = cols_to_fillna[i]
      train_cols.append(target)
    elif i == 2:
      train_cols = ['subcategory_num','caller_id_num','category_num']
      target = cols_to_fillna[i]
      train_cols.append(target)
    elif i == 3:
      train_cols = ['opened_by_num','sys_updated_by_num']
      target = cols_to_fillna[i]
      train_cols.append(target)
    else:
        break
    df_sub = numeric_df[train_cols].reset_index(drop=True)
    df_train = df_sub.dropna()
    df_test = df_sub[df_sub[cols_to_fillna[i]].isna()].drop(cols_to_fillna[i], axis=1)
    missing_indices = df_test.index
    imp_val = rmodel_acc(df_train,df_test,cols_to_fillna[i],0)
    imp_df.loc[missing_indices, cols_to_fillna[i]] = imp_val
    numeric_df[cols_to_fillna[i]] = imp_df[cols_to_fillna[i]]
    print(f"Filled na for {target}")
    #print(numeric_df.info())

for col in numeric_df.columns:
    df_cleaned[col] = numeric_df[col]
    d_miss = df_cleaned[df_cleaned[col].isna()][col]
    percent(d_miss,col,total_length)
print(df_cleaned.info())
df_cleaned.to_csv('IncidentFile_APP.csv',header=True)