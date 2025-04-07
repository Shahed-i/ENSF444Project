import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

def load_and_preprocess_data(filepath):
    # Load dataset
    df = pd.read_csv(filepath)

    # Encode target
    label_encoder = LabelEncoder()
    df['Drug'] = label_encoder.fit_transform(df['Drug'])

    # Features and target
    X = df.drop(columns=['Drug'])
    y = df['Drug']

    # Feature types
    categorical_features = ['Sex', 'BP', 'Cholesterol']
    numerical_features = ['Age', 'Na_to_K']

    # Column transformer
    column_transformer = ColumnTransformer([
        ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('scaler', StandardScaler(), numerical_features)
    ])

    X_transformed = column_transformer.fit_transform(X)

    return X_transformed, y, label_encoder, column_transformer
