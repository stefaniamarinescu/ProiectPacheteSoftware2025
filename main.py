
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

file_path = "medical_students_dataset.csv"
df_initial = pd.read_csv(file_path)
df = df_initial.copy()

def fill_missing_values(df):
    for col in df.columns:
        if df[col].isna().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
    return df

df_cleaned = fill_missing_values(df.copy())
df_cleaned['Diabetes'] = df_cleaned['Diabetes'].map({'Yes': 1, 'No': 0})
df_cleaned['Smoking'] = df_cleaned['Smoking'].map({'Yes': 1, 'No': 0})
df_cleaned['Gender'] = df_cleaned['Gender'].map({'Female': 0, 'Male': 1})
df_cleaned['Age'] = df_cleaned['Age'].round(0).astype(int)

categorical_cols = df_cleaned.select_dtypes(include=[object]).columns
df_encoded = pd.get_dummies(df_cleaned, columns=categorical_cols, drop_first=False, dtype=int)

numeric_cols_to_scale = ['Height', 'Weight', 'BMI', 'Temperature', 'Heart Rate', 'Blood Pressure', 'Cholesterol']

scaler_standard = StandardScaler()
df_standardized = df_encoded.copy()
df_standardized[numeric_cols_to_scale] = scaler_standard.fit_transform(df_standardized[numeric_cols_to_scale])
df_standardized[numeric_cols_to_scale] = df_standardized[numeric_cols_to_scale].round(2)

scaler_minmax = MinMaxScaler()
df_minmax = df_encoded.copy()
df_minmax[numeric_cols_to_scale] = scaler_minmax.fit_transform(df_minmax[numeric_cols_to_scale])
df_minmax[numeric_cols_to_scale] = df_minmax[numeric_cols_to_scale].round(2)

agg_df = df_cleaned.groupby(['Gender', 'Smoking'])[['BMI', 'Heart Rate', 'Blood Pressure', 'Cholesterol']].agg(['mean', 'sum', 'min', 'max', 'std']).round(2).reset_index()

stats = df_standardized.describe().round(2)
median_values = df_standardized.median(numeric_only=True).round(2)
sum_values = df_standardized.sum(numeric_only=True).round(2)

agg_diabetes = df_standardized[df_standardized['Diabetes'] == 1].groupby('Gender')[['BMI', 'Blood Pressure', 'Cholesterol']].agg(['mean', 'sum', 'min', 'max', 'std']).round(2)

bins = [0, 20, 30, 40, 100]
labels = ['<20', '20-30', '30-40', '40+']
df_standardized['Age Group'] = pd.cut(df_cleaned['Age'], bins=bins, labels=labels, right=False)
age_group_agg = df_standardized.groupby('Age Group')[['Heart Rate', 'Blood Pressure', 'Cholesterol']].agg(['mean', 'sum', 'count', 'std']).round(2)

def plot_outliers(df, columns):
    fig, axes = plt.subplots(nrows=len(columns), figsize=(10, 5 * len(columns)))
    if len(columns) == 1:
        axes = [axes]
    for ax, col in zip(axes, columns):
        sns.boxplot(x=df[col], ax=ax)
        ax.set_title(f"Boxplot pentru {col}")
    st.pyplot(fig)

st.set_page_config(page_title="Indicatori medicali", layout="wide")
st.markdown("<h1 style='text-align: center; color: black;'>Sănătatea studenților la Medicină: Analiza demografică și a semnelor vitale</h1>", unsafe_allow_html=True)
st.markdown("""
    <style>
    .stApp {
        background-color: #aa79ad;
    }
    .custom-title {
        color: #f2f1d0;
        font-size: 20px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("Navigați la:")
option = st.sidebar.radio("", ["Set de date", "Statistici descriptive", "Interacțiuni coloane DataFrame", "Histogramă", "Prelucrări avansate", "Vizualizare Outliers"])

if option == "Set de date":
    st.write("### Setul de date inițial")
    st.dataframe(df_initial.round(2), use_container_width=True)
    st.write("### După curățare și codificare")
    st.dataframe(df_encoded.round(2).head(10), use_container_width=True)
    st.write("### După standardizare")
    st.dataframe(df_standardized.round(2).head(10), use_container_width=True)
    st.write("### După MinMax scaling")
    st.dataframe(df_minmax.round(2).head(10), use_container_width=True)

if option == "Statistici descriptive":
    st.write("### Statistici descriptive (standardizat)")
    st.dataframe(stats, use_container_width=True)
    st.write("### Suma valorilor")
    st.dataframe(sum_values, use_container_width=True)
    st.write("### Mediana valorilor")
    st.dataframe(median_values, use_container_width=True)
    st.write("### Statistici descriptive")
    st.write(df.describe())
    st.write("### Primele 5 rânduri")
    st.write(df.head())

if option == "Interacțiuni coloane DataFrame":
    st.write("### Interacțiuni cu coloanele (standardizat)")

    if "df" not in st.session_state:
        st.session_state.df = df_standardized.copy()

    df = st.session_state.df

    st.write("### DataFrame curent:")
    st.dataframe(df)

    col_names = list(df.columns)
    if col_names:
        selected_col = st.selectbox("Selectează o coloană:", col_names)
    else:
        st.warning("Nu există coloane disponibile în DataFrame.")
        selected_col = None

    if selected_col:
        st.subheader(f"Operații pentru coloana: **{selected_col}**")

        new_name = st.text_input("Introdu noul nume pentru coloană:", value=selected_col)
        if st.button("Redenumește coloana"):
            if new_name and new_name != selected_col:
                st.session_state.df = st.session_state.df.rename(columns={selected_col: new_name})
                st.success(f"Coloana '{selected_col}' a fost redenumită în '{new_name}'!")
                st.rerun()
            else:
                st.info("Noul nume trebuie să fie diferit de cel curent.")

        if st.button("Șterge coloana"):
            st.session_state.df = st.session_state.df.drop(columns=[selected_col])
            st.success(f"Coloana '{selected_col}' a fost ștearsă!")
            st.rerun()

        if st.button("Afișează numărul de valori lipsă"):
            missing_count = st.session_state.df[selected_col].isna().sum()
            st.info(f"Coloana '{selected_col}' are {missing_count} valori lipsă.")

if option == "Histogramă":
    st.write("### Histogramă pentru date standardizate")
    numeric_columns = df_standardized.select_dtypes(include=[np.number]).columns.tolist()
    selected_column = st.selectbox("Alege o coloană numerică:", numeric_columns)
    hist_values = np.histogram(df_standardized[selected_column].dropna(), bins=10)[0]
    st.bar_chart(hist_values)

if option == "Prelucrări avansate":
    st.write("### Agregare pe gen și fumat")
    st.dataframe(agg_df, use_container_width=True)
    st.write("### Agregare pe diabet și gen")
    st.dataframe(agg_diabetes, use_container_width=True)
    st.write("### Agregare pe grupe de vârstă")
    st.dataframe(age_group_agg, use_container_width=True)

    st.write("### Alte prelucrări statistice")
    group_gender = df_standardized.groupby('Gender')[numeric_cols_to_scale].agg(['mean', 'std', 'min', 'max']).round(2)
    st.write("#### Indicatori statistici pe gen")
    st.dataframe(group_gender, use_container_width=True)

    group_smoking_age = df_standardized.groupby(['Smoking', 'Age Group'])[['BMI', 'Heart Rate']].agg(['mean', 'std', 'count']).round(2)
    st.write("#### Indicatori pe fumat și grupe de vârstă")
    st.dataframe(group_smoking_age, use_container_width=True)

    st.subheader("🔹 Clusterizare KMeans (scikit-learn)")
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    k = st.slider("Alege numărul de clustere", 2, 10, 3)
    pca = PCA(n_components=2)
    data_for_cluster = df_standardized[numeric_cols_to_scale]
    reduced_data = pca.fit_transform(data_for_cluster)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(reduced_data)
    cluster_labels = kmeans.labels_

    fig, ax = plt.subplots()
    scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_labels, cmap="viridis")
    ax.set_title("Rezultatul clusterizării (PCA + KMeans)")
    st.pyplot(fig)

    st.write(kmeans.cluster_centers_)

    st.subheader("🔹 Regresie logistică (scikit-learn)")
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    exclude_cols = ['Diabetes']
    if 'Age Group' in df_standardized.columns:
        exclude_cols.append('Age Group')

    X_logreg = df_standardized.drop(columns=exclude_cols)
    y_logreg = df_standardized['Diabetes']

    X_train, X_test, y_train, y_test = train_test_split(X_logreg, y_logreg, test_size=0.2, random_state=42)
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)

    st.text(classification_report(y_test, y_pred))

    st.subheader("🔹 Regresie liniară multiplă (statsmodels)")
    import statsmodels.api as sm

    X_multi = df_standardized[['Age', 'BMI', 'Cholesterol']]
    y_multi = df_standardized['Blood Pressure']
    X_multi = sm.add_constant(X_multi)

    model = sm.OLS(y_multi, X_multi).fit()
    st.text(model.summary())

if option == "Vizualizare Outliers":
    st.write("### Boxploturi pentru variabile standardizate")
    plot_outliers(df_standardized, numeric_cols_to_scale)
