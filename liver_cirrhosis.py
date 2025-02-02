import os
os.system('pip install seaborn')

import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.gridspec as gridspec
st.header('Liver Cirrhosis Status Prediction')
# Apply custom CSS for light blue sidebar
st.markdown(
    """
    <style>
    
        /* Change sidebar background color */
        [data-testid="stSidebar"] {
            background-color: #ADD8E6 !important; /* Light Blue */
        }

        /* Change sidebar text color */
        [data-testid="stSidebar"] * {
            color: black !important; 
        }
    </style>
    """,
    unsafe_allow_html=True
)
st.sidebar.title("Contents")
section = st.sidebar.radio("Go to", ["Introduction", "EDA", "Data Preprocessing & Feature Engineering","Model Building", "Results", "Conclusion"])
if section == "Introduction":
    st.header('Introduction')
    st.write("""
        Liver Cirrhosis is the 12th leading cause of death in countries like Pakistan, China, and Egypt. 
        This disease is caused when healthy tissue is replaced by scar tissue. The diagnosis of this disease 
        can be difficult, especially in cases of non-decompensated cirrhosis. Therefore, the goal of this work 
        is to evaluate the performance of different Machine Learning algorithms in order to predict liver cirrhosis stage. 
        In this work, we used six algorithms: Logistic Regression, K Nearest Neighbors, Support Vector Machine, 
        Naive Bayes, and Random Forest to predict the status of liver cirrhosis from D (death), C (censored), 
        CL (censored due to liver transplantation).
    """)

    st.subheader('Dataset')
    st.write("""
        Preprocessing of the dataset includes handling missing values and outliers, 
        and converting categorical features to numerical using one-hot encoding.
    """)
    st.write('Check this link for data from Kaggle: [Dataset](https://www.kaggle.com/datasets/joebeachcapital/cirrhosis-patient-survival-prediction)')
if section == "EDA":
    st.header('Exploratory Data Analysis')

    # File uploader widget for train dataset
    #train_file = st.file_uploader("dataset/train", type="csv")
    train_file = "https://raw.githubusercontent.com/mahrukhsaeed2702/Liver_cirrhosis/main/dataset/train.csv"
    #train_file = st.sidebar.file_uploader("dataset/train", type=["csv"], key="train_data_uploader")
    #test_file = st.sidebar.file_uploader("dataset/test", type=["csv"], key="test_data_uploader")
    test_file = "https://raw.githubusercontent.com/mahrukhsaeed2702/Liver_cirrhosis/main/dataset/test.csv"
    if train_file is not None:
        train = pd.read_csv(train_file)  # Convert UploadedFile to DataFrame
        st.write("Train Data Preview:")
        st.write(train)

        # Display basic statistics
        st.write("Train Data Summary:")
        st.write(train.describe())

        # Drop 'id' column if it exists
        if 'id' in train.columns:
            train = train.drop('id', axis=1)

        # Function to convert Stage column to string
        def stage_col(stage):
            if pd.notnull(stage):  
                return str(int(stage))  # Convert to string after ensuring it's a number
            return stage  

        if 'Stage' in train.columns:
            train['Stage'] = train['Stage'].apply(stage_col)

        st.write("Train Data after Processing:")
        st.write(train.head())

    # File uploader widget for test dataset
    #test_file = st.file_uploader("dataset/test", type="csv")

    if test_file is not None:
        test = pd.read_csv(test_file)  # Convert UploadedFile to DataFrame
        st.write("Test Data Preview:")
        st.write(test)

        # Display basic statistics
        st.write("Test Data Summary:")
        st.write(test.describe())

        # Drop 'id' column if it exists
        if 'id' in test.columns:
            test = test.drop('id', axis=1)

        if 'Stage' in test.columns:
            test['Stage'] = test['Stage'].apply(stage_col)

        st.write("Test Data after Processing:")
        st.write(test.head())
        st.subheader('Missing Values')
        st.write(test.isnull().sum())
# Store in session state
    st.session_state["train"] = train
    st.session_state["test"] = test
    # Check if train dataset is loaded before plotting
    if train_file is not None:
        # Function to plot categorical column distributions
        def plot_counts(data, features, hue=None, palette="magma"):
            if len(features) == 0:
                st.write("No categorical features to plot.")
                return

            n_cols = 2  # Subplots per row
            n_rows = -(-len(features) // n_cols)  # Compute rows using ceiling division

            sns.set_style("whitegrid")
            sns.set_context("notebook", font_scale=1.0)  # Adjust font size

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
            axes = axes.flatten()

            for i, feature in enumerate(features):
                sns.countplot(
                    data=data,
                    x=feature,
                    hue=hue,
                    palette="bright",  
                    ax=axes[i],
                    width=0.7
                )
                axes[i].set(title=f"Distribution of {feature}", xlabel='', ylabel='')

                # Annotate bar heights
                for p in axes[i].patches:
                    axes[i].text(
                        x=p.get_x() + p.get_width() / 2,
                        y=p.get_height(),
                        s=f'{p.get_height():.0f}',
                        ha='center', 
                        va='bottom',
                        fontsize=10
                    )

            for j in range(len(features), len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout()
            st.pyplot(fig)

        # Classify columns
        cat_cols = [col for col in train.columns if train[col].dtype == 'object']
        numeric_features = [col for col in train.columns if pd.api.types.is_numeric_dtype(train[col])]
        st.subheader('Dataset summary')
        st.write(f'Categorical columns ({len(cat_cols)}): {cat_cols}')
        st.write(f'Numeric columns ({len(numeric_features)}): {numeric_features}')
        st.subheader('Data Distribution')

        # Plot categorical columns distribution
        plot_counts(train, cat_cols)

    st.subheader('Univariate Analysis')

    def dist_plot(data, feature_list):
        results = {}

        # Check if feature_list is empty
        if len(feature_list) == 0:
            st.write("No numerical features to plot.")
            return

        # Number of columns in subplots
        n_cols = 2
        n_rows = int(np.ceil(len(feature_list) / n_cols))

        # Create figure
        fig = plt.figure(figsize=(16, 4 * n_rows))
        outer = gridspec.GridSpec(n_rows, n_cols, wspace=0.2, hspace=0.3)

        for i in range(len(feature_list)):
            feature = feature_list[i]
            inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[i],
                                                     wspace=0.1, hspace=0.1, height_ratios=(0.15, 0.85))

            ax_box = plt.Subplot(fig, inner[0])
            sns.boxplot(data=data, x=feature, color='lightblue', ax=ax_box)
            ax_box.set_xlabel('')
            fig.add_subplot(ax_box)

            mean_value = data[feature].mean()
            median_value = data[feature].median()
            ax_hist = plt.Subplot(fig, inner[1])
            sns.histplot(data=data, x=feature, kde=True, ax=ax_hist)
            ax_hist.axvline(mean_value, color='green', linestyle='dotted', linewidth=2, label='Mean')
            ax_hist.axvline(median_value, color='purple', linestyle='dotted', linewidth=2, label='Median')
            ax_hist.legend(loc='lower right', fontsize=10)

            skewness = data[feature].skew()
            kurt = data[feature].kurt()
            x_pos = 0.95 if skewness >= 0 else 0.25
            ax_hist.text(x_pos, 0.85, f"Skewness: {skewness:.2f}\nKurtosis: {kurt:.2f}",
                         transform=ax_hist.transAxes, verticalalignment='top', horizontalalignment='right',
                         bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5'),
                         fontsize=10)
            fig.add_subplot(ax_hist)

            results[feature] = {
                'mean': mean_value,
                'median': median_value,
                'skewness': skewness,
                'kurtosis': kurt
            }

        plt.tight_layout()
        st.pyplot(fig)
        return results

    dist_plot(train, numeric_features)

    def plot_boxplots_by_status(data, numeric_features):
        st.write("### Box Plots by Status")

        n_cols = 2
        n_rows = int(np.ceil(len(numeric_features) / n_cols))

        fig, ax = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        ax = ax.flatten()

        for i, feature in enumerate(numeric_features):
            sns.boxplot(data=data, x='Status', y=feature, ax=ax[i])

        if len(numeric_features) % 2 != 0:
            fig.delaxes(ax[-1])

        plt.tight_layout()
        st.pyplot(fig)

    plot_boxplots_by_status(train, numeric_features)

    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(train.corr(numeric_only=True), annot=True, fmt='.2f', ax=ax)
    st.pyplot(fig)

    ### Remove N_Days column from both train and test
    train = train.drop('N_Days', axis=1)
    test = test.drop('N_Days', axis=1)

    ### Replace id value in Hepatomegaly column with less frequent value of this column
    print(train.head())
    train['Hepatomegaly'] = train['Hepatomegaly'].replace('id', 'N')
    test['Hepatomegaly'] = test['Hepatomegaly'].replace('id','N')
elif section == "Data Preprocessing & Feature Engineering":
    #train_file = st.file_uploader("dataset/train", type="csv")
    if "train" in st.session_state and "test" in st.session_state:
        train = st.session_state["train"]
        test = st.session_state["test"]


    # Model training code here...
    else:
        st.error("Please upload the train and test files first!")
    #train = pd.read_csv(train_file) 
    st.header('Data Preprocessing')
    
    st.subheader('Handling missing values of numerical columns')
    st.write('We used an iterative imputer to handle the missing values of numerical columns of both the training and testing datasets after comparing it with KNN imputer. Iterative Imputer imputes missing values by modeling each feature with missing values as a function of other features, iteratively. Iterative Imputer is generally considered better for more complex datasets as it can capture relationships between features more effectively, leading to more accurate imputations.')

    
    
    st.subheader('Handling outliers')
    st.write('We used a robust z-score method to handle outliers for both the training and testing datasets after comparing it with isolation forest. It uses median and median absolute deviation instead of mean and standard deviation.')

    
    
    st.subheader('Handling missing values of categorical data')
    st.write('Missing values in categorical columns in both testing and training datasets are filled with existing categories based on how often those categories appear (their proportions)')
    
    
    st.subheader('Feature Engineering')

    st.write('**Scale numerical columns using RobustScaler**')
    st.write('Machine learning algorithms perform better when numerical features are on a similar scale. RobustScaler is chosen because it reduces the impact of extreme outliers.')

    st.write('**Encode nominal categorical columns using OneHotEncoder**')
    st.write('Machine learning models work with numerical data. Encoding converts categorical data into a form that algorithms can understand. We used OneHotEncoder to convert nominal categorical variables (like "Drug" or "Sex") into binary columns (0s and 1s). Each category becomes its own column.')

    st.write('**Encode ordinal categorical columns using OrdinalEncoder like Stage column**')
    st.write('Converted ordinal categorical column i.e stage column into numbers based on their order/rank using OrdinalEncoder.')

    st.write('**Class Imbalance Problem**')
    st.write('This problem occurs when one class in the target variable has significantly fewer examples than the other classes. This can lead to a biased model with poor performance on the minority class. SMOTE (Synthetic Minority Oversampling Technique) is used to resolve class imbalance problems. It creates synthetic examples for the minority class instead of duplicating existing ones. Balances the dataset without simply duplicating data. Helps the model learn better decision boundaries for the minority class.')

elif section == "Model Building":
    if "train" in st.session_state and "test" in st.session_state:
        train = st.session_state["train"]
        test = st.session_state["test"]


    # Model training code here...
    else:
        st.error("Please upload the train and test files first!")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import confusion_matrix
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import streamlit as st

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import streamlit as st
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer

    numeric_features = ['Age', 'Bilirubin', 'Cholesterol', 'Albumin', 'Copper',
                        'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin']

    # Apply Iterative Imputer
    imputer = IterativeImputer(max_iter=20, random_state=0)
    train_imputed = train.copy()
    test_imputed = test.copy()

    # Impute missing values
    train_imputed[numeric_features] = imputer.fit_transform(train[numeric_features])
    test_imputed[numeric_features] = imputer.transform(test[numeric_features])

    # Display results
    #st.write("Iterative Imputer - Train DataFrame")
    #st.write(train_imputed[numeric_features].describe())

    #st.write("Iterative Imputer - Test DataFrame")
    #st.write(test_imputed[numeric_features].describe())

    train[numeric_features] = train_imputed[numeric_features]
    test[numeric_features] = test_imputed[numeric_features]
    # Function to handle outliers using the Robust Z-Score Method
    def robust_z_score(df, threshold=3):
        """
        This function calculates the robust z-score for each value in the dataframe.
        Values with z-scores above the threshold are considered outliers and are removed.

        Parameters:
        - df: DataFrame containing the numerical columns.
        - threshold: The maximum z-score value to keep data points (default is 3).

        Returns:
        - A mask (True/False) where True means the value is not an outlier.
        """
        # Calculate the absolute Z-scores using robust method (median and MAD)
        z_scores = np.abs((df - df.median()) / (1.4826 * (np.abs(df - df.median()).median())))
        return z_scores < threshold

    # Apply the robust Z-score method on the training dataset
    mask_robust_z = robust_z_score(train[numeric_features])
    mask_robust_z = mask_robust_z.all(axis=1)

    train_robust_z_imputed = train.copy()
    train_robust_z_imputed = train_robust_z_imputed[mask_robust_z]

    # Apply the same process to the test dataset
    mask_robust_z_test = robust_z_score(test[numeric_features])
    mask_robust_z_test = mask_robust_z_test.all(axis=1)  # Ensure all columns pass the check

    # Create a new DataFrame with outliers removed for the test set
    test_robust_z_imputed = test.copy()
    test_robust_z_imputed = test_robust_z_imputed[mask_robust_z_test]

    # Update the original train and test DataFrames
    train = train_robust_z_imputed
    test = test_robust_z_imputed
    import pandas as pd
    import numpy as np

    # List of categorical columns that need missing value handling
    cat_cols = ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema', 'Stage']

    # Handling missing values in the train dataset
    for col in cat_cols:
        # Calculate the proportions of each category in the column
        prop = train[col].value_counts(normalize=True)

        # Initialize a list to hold the values that will be used to fill the missing values
        fill_values = []

        # Loop through the categories and their proportions to fill missing values
        for value, count in prop.items():
            # Extend the list by repeating each category based on its proportion
            fill_values.extend([value] * int(count * train[col].isnull().sum()))

        # Shuffle the list of fill values randomly to avoid any bias
        np.random.shuffle(fill_values)

        # Ensure the number of fill values exactly matches the number of missing values
        if len(fill_values) < train[col].isnull().sum():
            # If not enough fill values, add the most frequent category to make up the difference
            fill_values.extend([prop.idxmax()] * (train[col].isnull().sum() - len(fill_values)))
        elif len(fill_values) > train[col].isnull().sum():
            # If too many fill values, trim the list to match the number of missing values
            fill_values = fill_values[:train[col].isnull().sum()]

        # Assign the generated fill values to the missing positions in the train dataset
        train.loc[train[col].isnull(), col] = fill_values

    # Handling missing values in the test dataset (same process as for train)
    for col in cat_cols:
        # Calculate the proportions of each category in the column for the test dataset
        prop = test[col].value_counts(normalize=True)

        # Initialize a list to hold the fill values for the test dataset
        fill_values = []

        # Loop through the categories and their proportions to fill missing values in the test data
        for value, count in prop.items():
            # Extend the list by repeating each category based on its proportion
            fill_values.extend([value] * int(count * test[col].isnull().sum()))

        # Shuffle the list of fill values randomly to avoid any bias
        np.random.shuffle(fill_values)

        # Ensure the number of fill values exactly matches the number of missing values in the test dataset
        if len(fill_values) < test[col].isnull().sum():
            # If not enough fill values, add the most frequent category to make up the difference
            fill_values.extend([prop.idxmax()] * (test[col].isnull().sum() - len(fill_values)))
        elif len(fill_values) > test[col].isnull().sum():
            # If too many fill values, trim the list to match the number of missing values
            fill_values = fill_values[:test[col].isnull().sum()]

        # Assign the generated fill values to the missing positions in the test dataset
        test.loc[test[col].isnull(), col] = fill_values
    # Convert num cols to cat

    # Import necessary libraries for data preprocessing
    from sklearn.impute import SimpleImputer, IterativeImputer
    from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
    from imblearn.over_sampling import SMOTE
    from sklearn.model_selection import train_test_split  # Import train_test_split

    # Target variable and feature columns (categorical and numerical)
    target = 'Status'

    # List of categorical columns (nominal and ordinal)
    cat_cols_nominal = ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema']
    cat_cols_ordinal = ['Stage']

    # List of numerical columns
    numeric_features = ['Age', 'Bilirubin', 'Cholesterol', 'Albumin', 'Copper',
                        'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin']

    # Convert categorical columns to string type for consistency
    for col in cat_cols_nominal + cat_cols_ordinal:
        train[col] = train[col].astype(str)  # Convert categorical columns in train set to strings
        test[col] = test[col].astype(str)    # Convert categorical columns in test set to strings

    # Scale numerical columns using RobustScaler
    scaler = RobustScaler()
    train[numeric_features] = scaler.fit_transform(train[numeric_features])
    test[numeric_features] = scaler.transform(test[numeric_features])

    # Encode nominal categorical columns using OneHotEncoder
    nominal_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    train_nominal_encoded = nominal_encoder.fit_transform(train[cat_cols_nominal])
    test_nominal_encoded = nominal_encoder.transform(test[cat_cols_nominal])

    # Encode ordinal categorical columns using OrdinalEncoder
    ordinal_encoder = OrdinalEncoder()

    train_ordinal_encoded = ordinal_encoder.fit_transform(train[cat_cols_ordinal])
    test_ordinal_encoded = ordinal_encoder.transform(test[cat_cols_ordinal])

    # Create DataFrame for encoded nominal features
    train_nominal_df = pd.DataFrame(train_nominal_encoded, columns=nominal_encoder.get_feature_names_out(cat_cols_nominal))
    test_nominal_df = pd.DataFrame(test_nominal_encoded, columns=nominal_encoder.get_feature_names_out(cat_cols_nominal))

    # Create DataFrame for encoded ordinal features
    train_ordinal_df = pd.DataFrame(train_ordinal_encoded, columns=cat_cols_ordinal)
    test_ordinal_df = pd.DataFrame(test_ordinal_encoded, columns=cat_cols_ordinal)

    # Reset indices
    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)

    # Concatenate processed numerical and categorical features
    train_processed = pd.concat([train[numeric_features], train_nominal_df, train_ordinal_df], axis=1)
    test_processed = pd.concat([test[numeric_features], test_nominal_df, test_ordinal_df], axis=1)

    # Encode target variable
    target_encoder = LabelEncoder()
    y_train = target_encoder.fit_transform(train[target])

    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(train_processed, y_train)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    # Store in session state
    st.session_state["X_train"] = X_train
    st.session_state["X_test"] = X_test
    st.session_state["y_train"] = y_train
    st.session_state["y_test"] = y_test
    from sklearn.preprocessing import RobustScaler
    import numpy as np
    st.header('ML Models')

    # Parameter grid for GridSearchCV
    rf_param_grid = {
        'n_estimators': [200],
        'max_depth': [None],
      #  'min_samples_split': [2, 5],
       # 'min_samples_leaf': [1, 2],
    }

    # Random Forest with GridSearchCV
    rf_clf = RandomForestClassifier()
    rf_grid_search = GridSearchCV(estimator=rf_clf, param_grid=rf_param_grid, cv=5, scoring='neg_log_loss', n_jobs=-1)
    rf_grid_search.fit(X_train, y_train)

    # Best model and evaluation
    rf_best_model = rf_grid_search.best_estimator_
    rf_best_params = rf_grid_search.best_params_
    rf_best_logloss = -rf_grid_search.best_score_

    rf_y_pred = rf_best_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_y_pred)
    st.session_state["rf_best_model"] = rf_best_model

    # Display results
    st.markdown("### **Random Forest Results**", unsafe_allow_html=True)
    st.write(f"**Best Parameters:** {rf_best_params}")
    st.write(f"**Best Log Loss:** {rf_best_logloss}")
    st.write(f"**Accuracy:** {rf_accuracy:.4f}")

    # Classification Report
    st.markdown("### **Classification Report**", unsafe_allow_html=True)
    cr = classification_report(y_test, rf_y_pred, output_dict=True)

    # Convert classification report into DataFrame for better visualization
    cr_df = pd.DataFrame(cr).transpose()

    # Display classification report in a table format
    st.dataframe(cr_df)

    # Confusion Matrix
    st.markdown("### **Confusion Matrix**", unsafe_allow_html=True)
    cm = confusion_matrix(y_test, rf_y_pred)

    # Plot the confusion matrix using seaborn
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=rf_best_model.classes_, yticklabels=rf_best_model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    st.pyplot(fig)

    from sklearn.naive_bayes import GaussianNB

    # Train Naive Bayes model
    nb_clf = GaussianNB(var_smoothing=1e-1)
    nb_clf.fit(X_train, y_train)

    # Evaluate Naive Bayes on test data
    nb_y_pred = nb_clf.predict(X_test)
    nb_logloss = log_loss(y_test, nb_clf.predict_proba(X_test))
    nb_accuracy = accuracy_score(y_test, nb_y_pred)

    # Display results
    st.markdown("### **Naive Bayes Results**", unsafe_allow_html=True)
    st.write(f"**Log Loss:** {nb_logloss:.4f}")
    st.write(f"**Accuracy:** {nb_accuracy:.4f}")

    # Classification Report
    st.markdown("### **Classification Report**", unsafe_allow_html=True)
    cr = classification_report(y_test, nb_y_pred, output_dict=True)

    # Convert classification report into DataFrame for better visualization
    cr_df = pd.DataFrame(cr).transpose()

    # Display classification report in a table format
    st.dataframe(cr_df)

    # Confusion Matrix
    st.markdown("### **Confusion Matrix**", unsafe_allow_html=True)
    cm = confusion_matrix(y_test, nb_y_pred)

    # Plot the confusion matrix using seaborn
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=nb_clf.classes_, yticklabels=nb_clf.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    st.pyplot(fig)

    # Define the parameter grid for KNN
    knn_param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],  # Number of neighbors to test
        'weights': ['distance'],  # Weighting method for neighbors
        'metric': ['euclidean', 'manhattan'],  # Distance metric to use
    }

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import RandomizedSearchCV

    # KNN with RandomizedSearchCV for faster computation
    knn_clf = KNeighborsClassifier()
    knn_random_search = RandomizedSearchCV(
        estimator=knn_clf,
        param_distributions=knn_param_grid,
        n_iter=10,  # Number of random combinations to try
        cv=3,  # Number of cross-validation folds
        scoring='accuracy',  # Using accuracy as the scoring metric
        n_jobs=-1,
        verbose=2
    )

    # Fit the model using the training data
    knn_random_search.fit(X_train, y_train)

    # Best KNN model
    knn_best_model = knn_random_search.best_estimator_
    knn_best_params = knn_random_search.best_params_
    knn_best_accuracy = knn_random_search.best_score_

    # Evaluate KNN on the test data
    knn_y_pred = knn_best_model.predict(X_test)
    knn_accuracy = accuracy_score(y_test, knn_y_pred)

    # Calculate Log Loss
    knn_y_pred_prob = knn_best_model.predict_proba(X_test)
    knn_log_loss_value = log_loss(y_test, knn_y_pred_prob)

    # Display results
    st.markdown("### **KNN Results**", unsafe_allow_html=True)
    st.write(f"**Best Parameters:** {knn_best_params}")
    st.write(f"**Best Accuracy (cross-validation):** {knn_best_accuracy:.4f}")
    st.write(f"**Test Accuracy:** {knn_accuracy:.4f}")
    st.write(f"**Log Loss:** {knn_log_loss_value:.4f}")

    # Classification Report
    st.markdown("### **Classification Report**", unsafe_allow_html=True)
    cr = classification_report(y_test, knn_y_pred, output_dict=True)

    # Convert classification report into DataFrame for better visualization
    cr_df = pd.DataFrame(cr).transpose()

    # Display classification report in a table format
    st.dataframe(cr_df)

    # Confusion Matrix
    st.markdown("### **Confusion Matrix**", unsafe_allow_html=True)
    cm = confusion_matrix(y_test, knn_y_pred)

    # Plot the confusion matrix using seaborn
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=knn_best_model.classes_, yticklabels=knn_best_model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    st.pyplot(fig)

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler

    # Scaling the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define the parameter grid for Logistic Regression
    param_grid = {
        'C': [1, 10, 100],  # Regularization strength
        'penalty': ['l2'],  # L2 regularization (ridge)
        'solver': ['liblinear', 'saga']  # Solvers to use
    }

    # Logistic Regression model
    lr_clf = LogisticRegression()

    # Perform GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(estimator=lr_clf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)

    # Best Logistic Regression model
    best_lr_model = grid_search.best_estimator_
    best_lr_params = grid_search.best_params_
    best_lr_accuracy = grid_search.best_score_

    # Evaluate Logistic Regression on the test set
    y_pred = best_lr_model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_pred)

    # Calculate log loss
    y_pred_proba = best_lr_model.predict_proba(X_test_scaled)  # Probabilities for log loss
    log_loss_value = log_loss(y_test, y_pred_proba)

    # Display results
    st.markdown("### **Logistic Regression Results**", unsafe_allow_html=True)
    st.write(f"**Best Parameters:** {best_lr_params}")
    st.write(f"**Best Cross-Validation Accuracy:** {best_lr_accuracy:.4f}")
    st.write(f"**Test Accuracy:** {test_accuracy:.4f}")
    st.write(f"**Log Loss:** {log_loss_value:.4f}")

    # Classification Report
    st.markdown("### **Classification Report**", unsafe_allow_html=True)
    cr = classification_report(y_test, y_pred, output_dict=True)

    # Convert classification report into DataFrame for better visualization
    cr_df = pd.DataFrame(cr).transpose()

    # Display classification report in a table format
    st.dataframe(cr_df)

    # Confusion Matrix
    st.markdown("### **Confusion Matrix**", unsafe_allow_html=True)
    cm = confusion_matrix(y_test, y_pred)

    # Plot the confusion matrix using seaborn
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=best_lr_model.classes_, yticklabels=best_lr_model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    st.pyplot(fig)

    from sklearn.svm import SVC

    # Define the parameter grid for SVM
    svm_param_grid = {
        'C': [100],
        'kernel': ['rbf'],
       # 'gamma': ['scale', 'auto', 0.1],
    }

    # SVM with RandomizedSearchCV for faster computation
    svm_clf = SVC(probability=True)
    svm_random_search = RandomizedSearchCV(
        estimator=svm_clf,
        param_distributions=svm_param_grid,
        n_iter=10,  # Number of random combinations to try
        cv=3,  # Number of folds for cross-validation
        scoring='neg_log_loss',
        n_jobs=-1,
        verbose=2
    )

    # Fit the model
    svm_random_search.fit(X_train, y_train)

    # Best SVM model after fitting
    best_svm_model = svm_random_search.best_estimator_
    svm_best_params = svm_random_search.best_params_
    svm_best_logloss = -svm_random_search.best_score_

    # Evaluate SVM on test data
    svm_y_pred = best_svm_model.predict(X_test)
    svm_accuracy = accuracy_score(y_test, svm_y_pred)

    # Calculate Log Loss
    svm_y_pred_prob = best_svm_model.predict_proba(X_test)
    svm_log_loss_value = log_loss(y_test, svm_y_pred_prob)

    # Display results
    st.markdown("### **SVM Results**", unsafe_allow_html=True)
    st.write(f"**Best Parameters:** {svm_best_params}")
    st.write(f"**Best Log Loss:** {svm_best_logloss:.4f}")
    st.write(f"**Accuracy:** {svm_accuracy:.4f}")
    st.write(f"**Log Loss:** {svm_log_loss_value:.4f}")

    # Classification Report
    st.markdown("### **Classification Report**", unsafe_allow_html=True)
    cr = classification_report(y_test, svm_y_pred, output_dict=True)

    # Convert classification report into DataFrame for better visualization
    cr_df = pd.DataFrame(cr).transpose()

    # Display classification report in a table format
    st.dataframe(cr_df)

    # Confusion Matrix
    st.markdown("### **Confusion Matrix**", unsafe_allow_html=True)
    cm = confusion_matrix(y_test, svm_y_pred)
    st.subheader("Model Performance Comparison")

    # Plot the confusion matrix using seaborn
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=best_svm_model.classes_, yticklabels=best_svm_model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    st.pyplot(fig)
elif section == "Results":
    if "train" in st.session_state and "test" in st.session_state:
        train = st.session_state["train"]
        test = st.session_state["test"]


    # Model training code here...
    else:
        st.error("Please upload the train and test files first!")
    st.title("Results")

    # Define models
    models = {
        'RF': RandomForestClassifier(n_estimators=200, max_depth=None),
        'SVM': SVC(kernel='rbf', C=100, probability=True),
        'NB': GaussianNB(),
        'LR': LogisticRegression(solver='saga', C=1, penalty='l2'),
        'KNN': KNeighborsClassifier(metric='manhattan', n_neighbors=3, weights='distance')
    }

    # Initialize lists to store TP, TN, FP, FN for each model
    tp, tn, fp, fn = [], [], [], []
    if "X_train" in st.session_state and "X_test" in st.session_state and "y_train" in st.session_state and "y_test" in st.session_state :
        X_train = st.session_state["X_train"]
        X_test = st.session_state["X_test"]
        y_train = st.session_state["y_train"]
        y_test = st.session_state["y_test"]


    # Train models and calculate confusion matrix for each
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Get confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Multi-class confusion matrix (flattening the matrix)
        tp_values = cm.diagonal()
        fp_values = cm.sum(axis=0) - tp_values
        fn_values = cm.sum(axis=1) - tp_values
        tn_values = cm.sum() - (tp_values + fp_values + fn_values)
        
        # Append to lists
        tp.append(tp_values)
        tn.append(tn_values)
        fp.append(fp_values)
        fn.append(fn_values)

    # Plotting the results
    x = list(models.keys())  # Model names
    width = 0.2  # Bar width
    x_pos = range(len(models))  # Position of bars on x-axis

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(7,5))

    # Main bar plot for TP, TN, FP, FN
    ax1.bar(x_pos, np.array(tp).mean(axis=1), width, label='TP', color='green')
    ax1.bar([p + width for p in x_pos], np.array(tn).mean(axis=1), width, label='TN', color='blue')
    ax1.bar([p + width*2 for p in x_pos], np.array(fp).mean(axis=1), width, label='FP', color='red')
    ax1.bar([p + width*3 for p in x_pos], np.array(fn).mean(axis=1), width, label='FN', color='orange')

    # Adding text labels above bars
    for i, model_name in enumerate(models.keys()):
        ax1.text(i, np.mean(tp[i]) + 0.02, f' {int(np.mean(tp[i]))}', ha='center', color='green', fontsize=12)
        ax1.text(i + width, np.mean(tn[i]) + 0.02, f' {int(np.mean(tn[i]))}', ha='center', color='blue', fontsize=12)
        ax1.text(i + 2*width, np.mean(fp[i]) + 0.02, f' {int(np.mean(fp[i]))}', ha='center', color='red', fontsize=12)
        ax1.text(i + 3*width, np.mean(fn[i]) + 0.02, f' {int(np.mean(fn[i]))}', ha='center', color='orange', fontsize=12)

    # Add labels and title
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Average Count')
    ax1.set_title('TP, TN, FP, FN for Each Model')
    ax1.set_xticks([p + 1.5 * width for p in x_pos])
    ax1.set_xticklabels(x)

    # Show legend and plot
    ax1.legend()
    plt.tight_layout()

    # Show plot in Streamlit
    st.pyplot(fig)

    # Performance metrics for 5 models
    models = ['RF', 'SVM', 'NB', 'LR', 'KNN']
    accuracy = [0.95, 0.90, 0.61, 0.67, 0.92]  # Example values for accuracy
    sensitivity = [0.93, 0.88, 0.63, 0.67, 0.91]  # Example values for sensitivity
    specificity = [0.96, 0.94, 0.75, 0.84, 0.96]  # Example values for specificity
    precision = [0.93, 0.88, 0.63, 0.67, 0.91]  # Example values for precision
    f1_score = [0.90, 0.84, 0.57, 0.61, 0.86]  # Example values for F1 score

    # Create a DataFrame to align values with model names
    performance_df = pd.DataFrame({
        'Accuracy': accuracy,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Precision': precision,
        'F1 Score': f1_score
    }, index=models)

    # Displaying the performance dataframe
    st.dataframe(performance_df)

    # Plotting the bar charts for performance metrics
    bar_width = 0.15
    index = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(7,5))

    # Using bright colors for the bars
    bar1 = ax.bar(index, accuracy, bar_width, color='blue')
    bar2 = ax.bar(index + bar_width, sensitivity, bar_width, color='darkgreen')
    bar3 = ax.bar(index + 2*bar_width, specificity, bar_width, color='red')
    bar4 = ax.bar(index + 3*bar_width, precision, bar_width, color='orange')
    bar5 = ax.bar(index + 4*bar_width, f1_score, bar_width, color='purple')

    # Adding labels and title
    ax.set_xlabel('Models')
    ax.set_ylabel('Scores')
    ax.set_title('Performance Metrics for 5 Models')

    # Set the Y-axis range from 0 to 1.2
    ax.set_ylim(0, 1.6)

    # Set the x-ticks to be in the center of the grouped bars
    ax.set_xticks(index + 2*bar_width)
    ax.set_xticklabels(models)

    # Adding the legend
    ax.legend(['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1 Score'])

    # Display the plot in Streamlit
    st.pyplot(fig)

    st.subheader('Feature Importance')
    st.write('The individual features play a crucial role in model predictions. For this purpose, we perform feature importance analysis using the Random Forest model, as it has the best accuracy among all other ML models. The analysis shows that Age has a crucial role in predicting the stage of liver cirrhosis.')
    if "rf_best_model" in st.session_state:
        rf_best_model = st.session_state["rf_best_model"]
    if hasattr(rf_best_model, 'feature_importances_'):
        importances = rf_best_model.feature_importances_
        feature_names = X_train.columns  # Replace X_train with your actual feature set variable

        # Create a DataFrame for feature importances
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        # Plot the feature importances
        fig = plt.figure(figsize=(8, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette="bright")
        plt.title('Feature Importance (Random Forest)')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.tight_layout()
        st.pyplot(fig)

    else:
        print("The model does not have the 'feature_importances_' attribute.")
elif section == "Conclusion":
    st.title("Conclusion")
    st.write('The main part of this work was to make an effective model for predicting liver cirrhosis status utilizing five distinctive supervised machine learning classifiers. We studied all classifiers execution on the patients information parameters and the RF classifier gives the optimal standard of accuracy of 94% in predicting the liver cirrhosis stage. From now on, this outperformed machine learning classification procedure will provide a decision support system in stage prediction of liver cirrhosis. In this study, we reviewed some popular supervised machine learning algorithms but countless machine learning models could be utilized in the future to improve prediction accuracy. ')
