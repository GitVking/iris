import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

st.title('鸢尾花数据集分析')
'''
>1, 鸢尾花数据集介绍  
>2, 鸢尾花数据可视化探索  
>3, 模型建立及品种预测
'''

source_image = "iris.png"
source_csv = "iris.csv"

@st.cache(suppress_st_warning=True)
def get_data(source_image, source_csv):
    image = Image.open(source_image)
    df = pd.read_csv(source_csv)
    st.write("cache miss")
    return image, df

image, df = get_data(source_image, source_csv)

# if st.checkbox('鸢尾花数据集介绍'):
'''
(Fisher's) Iris鸢尾花数据集是英国统计学家、优生学家和生物学家Ronald Fisher在1936年发表的论文*The use of multiple measurements in taxonomic problems*作为线性判别分析的一个例子而引入的一个**多变量**数据集。它有时被称为Anderson's Iris data set , 因为Edgar Anderson收集的数据是为了*量化三个相关物种的鸢尾花的形态变异*。  
该数据集包括来自三个鸢尾花品种（**Iris setosa**、**Iris virginica**和**Iris versicolor**）的各*50*个样本，以及**萼片长度**、**萼片宽度**、**花瓣长度**、**花瓣宽度**和**品种**等五个属性下的**150**条记录。从每个样本中测量出四个特征：萼片和花瓣的长度和宽度，单位为厘米。   
>花萼是一朵花中所有萼片的总称，包被在花的最外层。萼片一般呈绿色的叶片状，其形态和构造与叶片相似。  
>iris:鸢尾花  sepal:花萼  petal:花瓣 
'''
st.image(image, caption='鸢尾花图解')

if st.checkbox('鸢尾花数据可视化探索 '):
    if st.checkbox('源数据'):
        st.write(df)
    st.write('在左侧可以对数据进行筛选')

    '''
    ## 散点图  
    >使用散点图展示单个或多个鸢尾花品种在一个或两个方向上的特征  
    '''

    species = st.sidebar.multiselect('鸢尾花品种选择', df['variety'].unique())
    if not species:
        st.sidebar.error("请至少选择一个鸢尾花品种")
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
        if new_df.empty:
            st.error('请在左侧选择品种')
        else:
            st.success('以上为你选择品种的相关数据')

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
        if new_df.empty:
            st.error('请在左侧选择品种')
        else:
            st.success('以上为你选择品种的相关数据')



if st.checkbox('模型建立及品种预测'):
    features = df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']].values
    labels = df['variety'].values

    X_train, X_test, y_train, y_test = train_test_split(features, labels, train_size=0.7, random_state=1)



    alg = ['Support Vector Machine', 'Decision Tree']
    classifier = st.selectbox('算法选择', alg)
    if classifier == 'Decision Tree':
        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        print('%.2f%%' % (acc * 100))
        st.write('准确率: ', '%.2f%%' % (acc * 100))
        # st.write('准确率: ', acc)
        pred_clf = clf.predict(X_test)
        # cm_clf=confusion_matrix(y_test,pred_clf)
        # st.write('Confusion matrix: ', cm_clf)
        # st.write('混淆矩阵: ', cm_clf)
        # feature_input = st.text_input("请输入4个以逗号分割的特征值")
        # ans = feature_input.split(',')
        # for i in range(len(ans)):
        #     ans[i] = float(ans[i])
        # feature_value = np.array([(ans)])
        # if feature_value:
        # feature_prdt = clf.predict(feature_value)
        # emoji = ":blue_heart:" if feature_prdt == 'Setosa' else (
        #     ":green_heart:" if feature_prdt == 'Virginica' else ":heart:")
        # st.write('预测结果为: ', feature_prdt[0], emoji)

    elif classifier == 'Support Vector Machine':
        clf = SVC()
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        st.write('准确率: ', '%.2f%%' % (acc * 100))
        # st.write('准确率: ', acc)
        pred_clf = clf.predict(X_test)
        # cm=confusion_matrix(y_test,pred_clf)
        # st.write('Confusion matrix: ', cm)
        # st.write('混淆矩阵: ', cm)
        # feature_input = st.text_input("请输入4个以逗号分割的特征值")
        # ans = feature_input.split(',')
        # for i in range(len(ans)):
        #     ans[i] = float(ans[i])
        # feature_value = np.array([(ans)])
        # if feature_value:
        # feature_prdt = clf.predict(feature_value)[0]
        # emoji = ":blue_heart:" if feature_prdt == 'Setosa' else (
        #     ":green_heart:" if feature_prdt == 'Virginica' else ":heart:")
        # st.write('预测结果为: ', feature_prdt, emoji)

    sepal_length = st.slider("请输入花瓣长度:", min_value=0.0, max_value=10.0, value=5.1, step=0.1, help="可根据上面的可视化特征选择数据")
    sepal_width = st.slider("请输入花萼宽度:", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
    petal_length = st.slider("请输入花瓣长度:", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
    petal_width = st.slider("请输入花瓣宽度:", min_value=0.0, max_value=10.0, value=0.2, step=0.1)
    feature_value = np.array([(sepal_length, sepal_width, petal_length, petal_width)])
    feature_prdt = clf.predict(feature_value)
    emoji = ":blue_heart:" if feature_prdt == 'Setosa' else (
        ":green_heart:" if feature_prdt == 'Virginica' else ":heart:")
    '你选择的算法为', classifier
    st.write(f'预测结果为 -> *{feature_prdt[0]}* {emoji}')