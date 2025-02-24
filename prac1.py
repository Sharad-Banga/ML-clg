data = {
'Age': [25, 30, 35, None, 40, 45, 50, None],
'Gender': ['Male', 'Female', 'Female', 'Male', 'Female', 'Male',
'Female', 'Male'],
'Salary': [50000, 60000, 70000, 80000, 90000, None, 110000, 120000],
'Purchased': ['Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes']
}
df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)

# For numerical columns (Age and Salary), we'll replace missing values with
the mean
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
numerical_imputer = SimpleImputer(strategy='mean')
df[numerical_cols] = numerical_imputer.fit_transform(df[numerical_cols])
# For categorical columns (Gender and Purchased), we'll replace missing
values with the most frequent value
categorical_cols = df.select_dtypes(include=['object']).columns
categorical_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_cols] =
categorical_imputer.fit_transform(df[categorical_cols])
label_encoder = LabelEncoder()
for col in categorical_cols:
df[col] = label_encoder.fit_transform(df[col])

scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

X = df.drop('Purchased', axis=1) # Features
y = df['Purchased'] # Target column
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)

print("\nPreprocessed DataFrame:")
print(df)
print("\nTraining Data (X_train):")
print(X_train)
print("\nTesting Data (X_test):")
print(X_test)
print("\nTraining Labels (y_train):")
print(y_train)
print("\nTesting Labels (y_test):")
print(y_test)

