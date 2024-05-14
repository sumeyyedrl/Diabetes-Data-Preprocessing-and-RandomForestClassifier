import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


df=pd.read_csv("diabetes/diabetes.csv")
df.head()
df.describe([0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95]).T
df.isnull().sum()
df.isnull().sum().sum()

df.columns = [col.upper() for col in df.columns]

def check_corelation(dataframe):
    dataframe.corr()
    f, ax = plt.subplots(figsize=[18, 13])
    sns.heatmap(dataframe.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
    ax.set_title("Correlation Matrix", fontsize=20)
    plt.show(block=True)
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car
cat_cols, num_cols, cat_but_car = grab_col_names(df)

def cat_summary(dataframe, categorical_col, plot=False):
    print(pd.DataFrame({categorical_col: dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[categorical_col], data=dataframe)
        plt.show()
for col in cat_cols:
    cat_summary(df,col)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()
for col in num_cols:
    num_summary(df, col)

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")
for col in num_cols:
    target_summary_with_num(df, "OUTCOME", col)

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
def check_outlier(dataframe, col_name,q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name,q1,q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False
def grab_outliers(dataframe, col_name, index=False,q1=0.25, q3=0.75):
    low, up = outlier_thresholds(dataframe, col_name,q1,q3)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

def remove_outlier(dataframe, col_name,q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name,q1,q3)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers

def replace_with_thresholds(dataframe, variable,q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, variable,q1,q3)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    if(check_outlier(df,col)):
        print(col)
        grab_outliers(df,col)
        print("\n")

for col in num_cols:
    print(col, ":", df[df[col]==0].shape[0])

zero_columns = [col for col in df.columns if (df[col].min() == 0 and col not in ["PREGNANCIES", "OUTCOME"])]
df[df[zero_columns]==0] = np.NaN
df.isnull().sum()

check_corelation(df)

df["GLUCOSE"].fillna(df.groupby("OUTCOME")["GLUCOSE"].transform("median"),inplace=True)
df["BMI"].fillna(df.groupby("OUTCOME")["BMI"].transform("median"),inplace=True)


#GLUCOSE vs INSULIN
df["NEW_GLUCOSE_CAT"]=pd.cut(df["GLUCOSE"],[0,100,140,df["GLUCOSE"].max()], labels=[0,1,2])
df.groupby("NEW_GLUCOSE_CAT").agg({"INSULIN":"median"})
df["INSULIN"].fillna(df.groupby(["NEW_GLUCOSE_CAT","OUTCOME"])["INSULIN"].transform("median"),inplace=True)

#BMI vs SKINTHICKNESS
df["NEW_BMI_CAT"]=pd.cut(df["BMI"],[0,18,25,30,df["BMI"].max()], labels=[0,1,2,3])
df.groupby("NEW_BMI_CAT").agg({"SKINTHICKNESS":"median"})
df["SKINTHICKNESS"].fillna(df.groupby(["NEW_GLUCOSE_CAT","OUTCOME"])["SKINTHICKNESS"].transform("median"),inplace=True)

#BLOODPRESSURE vs AGE-GLUCOSE
df["NEW_AGE_CAT"]=pd.cut(df["AGE"],[0,16,30,45,60,df["AGE"].max()], labels=[0,1,2,3,4])
df.groupby(["NEW_GLUCOSE_CAT","NEW_AGE_CAT"]).agg({"BLOODPRESSURE":"median"})
df["BLOODPRESSURE"].fillna(df.groupby(["NEW_GLUCOSE_CAT","NEW_AGE_CAT","OUTCOME"])["SKINTHICKNESS"].transform("median"),inplace=True)

clf= LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)
df_scores=clf.negative_outlier_factor_
np.sort(df_scores)[0:5]
scores=pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True,xlim=[0,50],style=".-") #kırılımın olduğu yeri threshold olarak seçmek!
plt.show(block=True)
th = np.sort(df_scores)[4]

df[df_scores<th]
df[df_scores<th].shape
df.describe().T
df[df_scores<th].index
df[df_scores<th].drop(axis=0,labels=df[df_scores<th].index)



outlier_col_name=[]
for col in num_cols:
    if(check_outlier(df,col,0.05,0.95)):
        print(col)
        outlier_col_name.append(col)
        grab_outliers(df,col,0.05,0.95)
        print("\n")


for col in outlier_col_name:
    df=remove_outlier(df,col,0.05,0.95)


df["NEW_INSULIN_CAT"]=pd.cut(df["INSULIN"],[0,100,140,df["INSULIN"].max()], labels=["Low","Normal","High"])
df["NEW_BLOODPRESSURE_CAT"]=pd.cut(df["BLOODPRESSURE"],[0,80,90,df["BLOODPRESSURE"].max()], labels=["Low","Normal","High"])


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe
binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["OUTCOME"]]
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
df = one_hot_encoder(df, cat_cols, drop_first=True)

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

y = df["OUTCOME"]
X = df.drop("OUTCOME", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred,y_test),3)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}")
print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}")


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    print(feature_imp.sort_values("Value",ascending=False))
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X)