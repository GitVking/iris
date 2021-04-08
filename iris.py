import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from PIL import Image

st.title('鸢尾花数据集分析')
source_image = "iris.png"
source_csv = "iris.csv"


@st.cache(suppress_st_warning=True)
def get_data(source_image, source_csv):
    image = Image.open(source_image)
    df = pd.read_csv(source_csv)
    st.write("cache miss")
    return image, df


image, df = get_data(source_image, source_csv)
if st.checkbox('鸢尾花介绍'):
    '''
    iris:鸢尾花  
    petal:花瓣  
    sepal:花萼    
    >花萼是一朵花中所有萼片的总称，包被在花的最外层。萼片一般呈绿色的叶片状，其形态和构造与叶片相似。  
    >该数据集中的鸢尾花种类分为*Setosa*,*Versicolor*,*Virginica*三种
    '''
    st.image(image, caption='鸢尾花图解')
if st.checkbox('全部数据'):
    st.write(df)

# st.subheader('散点图')
'''
## 散点图  
>使用散点图展示单个或多个鸢尾花品种在一个或两个方向上的特征  
'''

species = st.sidebar.multiselect('鸢尾花品种选择', df['variety'].unique())
if not species:
    st.sidebar.error("请至少选择一个鸢尾花品种")
# st.sidebar.success("请至少选择一个鸢尾花品种")
col1 = st.sidebar.selectbox('x轴特征选择', df.columns[0:4])
col2 = st.sidebar.selectbox('y轴特征选择', df.columns[0:4])

new_df = df[(df['variety'].isin(species))]

# create figure using plotly express
if species:
    fig = px.scatter(new_df, x=col1, y=col2, color='variety')
    # Plot!
    st.plotly_chart(fig)
else:
    st.write()
if st.checkbox('散点图数据'):
    st.write(new_df)

# st.subheader('柱状图')
'''
## 柱状图  
>使用柱状图比较单个或多个鸢尾花品种的某一特征  
'''
feature = st.selectbox('特征选择', df.columns[0:4])
# Filter dataframe
new_df2 = df[(df['variety'].isin(species))][feature]
if species:
    fig2 = px.histogram(new_df, x=feature, color="variety", marginal="rug")
    st.plotly_chart(fig2)
if st.checkbox('柱状图数据'):
    st.write(new_df2)
st.subheader('Machine Learning models')

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

features = df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']].values
labels = df['variety'].values

X_train, X_test, y_train, y_test = train_test_split(features, labels, train_size=0.7, random_state=1)

sepal_length = st.slider("请输入花瓣长度:", min_value=0.0, max_value=10.0, value=5.1, step=0.1, help="可根据上面的可视化特征选择数据")
sepal_width = st.slider("请输入花萼宽度:", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
petal_length = st.slider("请输入花瓣长度:", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
petal_width = st.slider("请输入花瓣宽度:", min_value=0.0, max_value=10.0, value=0.2, step=0.1)
feature_value = np.array([(sepal_length, sepal_width, petal_length, petal_width)])

alg = ['Support Vector Machine', 'Decision Tree']
classifier = st.sidebar.selectbox('算法选择', alg)
if classifier == 'Decision Tree':
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    acc = dtc.score(X_test, y_test)
    print('%.2f%%' % (acc * 100))
    st.write('准确率: ', '%.2f%%' % (acc * 100))
    # st.write('准确率: ', acc)
    pred_dtc = dtc.predict(X_test)
    # cm_dtc=confusion_matrix(y_test,pred_dtc)
    # st.write('Confusion matrix: ', cm_dtc)
    # st.write('混淆矩阵: ', cm_dtc)
    # feature_input = st.text_input("请输入4个以逗号分割的特征值")
    # ans = feature_input.split(',')
    # for i in range(len(ans)):
    #     ans[i] = float(ans[i])
    # feature_value = np.array([(ans)])
    # if feature_value:
    feature_prdt = dtc.predict(feature_value)
    emoji = ":blue_heart:" if feature_prdt == 'Setosa' else (
        ":green_heart:" if feature_prdt == 'Virginica' else ":heart:")
    st.write('预测结果为: ', feature_prdt[0], emoji)

elif classifier == 'Support Vector Machine':
    svm = SVC()
    svm.fit(X_train, y_train)
    acc = svm.score(X_test, y_test)
    st.write('准确率: ', '%.2f%%' % (acc * 100))
    # st.write('准确率: ', acc)
    pred_svm = svm.predict(X_test)
    # cm=confusion_matrix(y_test,pred_svm)
    # st.write('Confusion matrix: ', cm)
    # st.write('混淆矩阵: ', cm)
    # feature_input = st.text_input("请输入4个以逗号分割的特征值")
    # ans = feature_input.split(',')
    # for i in range(len(ans)):
    #     ans[i] = float(ans[i])
    # feature_value = np.array([(ans)])
    # if feature_value:
    feature_prdt = svm.predict(feature_value)[0]
    emoji = ":blue_heart:" if feature_prdt == 'Setosa' else (
        ":green_heart:" if feature_prdt == 'Virginica' else ":heart:")
    st.write('预测结果为: ', feature_prdt, emoji)
