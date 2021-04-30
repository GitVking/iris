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
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import metrics
from hulearn.experimental.interactive import InteractiveCharts
import asyncio
def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()
get_or_create_eventloop()
st.title('鸢尾花数据集分析')
# '''
# >1, 鸢尾花数据集介绍  
# >2, 鸢尾花数据可视化探索  
# >3, 模型建立及品种预测
# '''

source_image = "./material/iris.png"
source_csv = "./material/iris.csv"

@st.cache(suppress_st_warning=True)
def get_data(source_image, source_csv):
    image = Image.open(source_image)
    df = pd.read_csv(source_csv)
    st.write("cache miss")
    return image, df

image, df = get_data(source_image, source_csv)

with st.beta_expander('鸢尾花数据集介绍', expanded=True):
    '''
    (Fisher's) Iris鸢尾花数据集是英国统计学家、优生学家和生物学家Ronald Fisher在1936年发表的论文*The use of multiple measurements in taxonomic problems*作为线性判别分析的一个例子而引入的一个**多变量**数据集。它有时被称为Anderson's Iris data set , 因为Edgar Anderson收集的数据是为了*量化三个相关物种的鸢尾花的形态变异*。  
    该数据集包括来自三个鸢尾花品种（**Iris setosa**、**Iris virginica**和**Iris versicolor**）的各*50*个样本，以及**萼片长度**、**萼片宽度**、**花瓣长度**、**花瓣宽度**和**品种**等五个属性下的**150**条记录。从每个样本中测量出四个特征：萼片和花瓣的长度和宽度，单位为厘米。   
    >花萼是一朵花中所有萼片的总称，包被在花的最外层。萼片一般呈绿色的叶片状，其形态和构造与叶片相似。  
    >iris:鸢尾花  sepal:花萼  petal:花瓣 
    '''
    st.image(image, caption='鸢尾花图解')

with st.beta_expander('鸢尾花数据可视化探索 '):
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



with st.beta_expander('模型建立及品种预测'):
    features = df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']].values
    labels = df['variety'].values

    X_train, X_test, y_train, y_test = train_test_split(features, labels, train_size=0.7, random_state=1)
    train = pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train)], axis=1)
    train.columns = ['sepal_length','sepal_width','petal_length','petal_width', 'variety']
    # if st.checkbox('data'):
        # st.write(train)



    alg = ['Support Vector Machine', 'KNN','Decision Tree']
    classifier = st.sidebar.selectbox('算法选择', alg)
    if classifier == 'Decision Tree':
        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        pred_clf = clf.predict(X_test)


    elif classifier == 'Support Vector Machine':
        clf = SVC()
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        pred_clf = clf.predict(X_test)

        
    elif classifier == 'KNN':
        if st.checkbox('算法介绍', value = True):
            '''
            最近邻居法（KNN算法，又译K-近邻算法）是一种用于分类和回归的非参数统计方法。  
            >在k-NN分类中，输出是一个分类族群。一个对象的分类是由其邻居的“多数表决”确定的，k个最近邻居（k为正整数，通常较小）中最常见的分类决定了赋予该对象的类别。若k = 1，则该对象的类别直接由最近的一个节点赋予。
            '''
            k_range = range(1,100)
            scores = {}
            scores_list = []
            for k in k_range:
                clf = KNeighborsClassifier(n_neighbors=k)
                clf.fit(X_train, y_train)
                pred_clf = clf.predict(X_test)
                scores[k] = metrics.accuracy_score(y_test,pred_clf)
                scores_list.append(metrics.accuracy_score(y_test,pred_clf))
            # st.line_chart(scores_list)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.plot(k_range,scores_list)   
            # plt.xlabel('k_nearest_neighbors)')
            #plt.ylabel('accuracy_score')
            plt.xlabel('邻居数(K)',fontproperties = 'SimHei')
            plt.ylabel('准确度',fontproperties = 'SimHei')
            st.pyplot()
        k_value = st.number_input("请输入算法的邻居数(k)", value = 55, step = 5)
        clf = KNeighborsClassifier(n_neighbors=k_value)
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        pred_clf = clf.predict(X_test)

        
    '''**分类器预测**'''
    with st.form(key='my_form'):
        sepal_length = st.slider("请输入花瓣长度:", min_value=0.0, max_value=10.0, value=5.1, step=0.1, help="可根据上面的可视化特征选择数据")
        sepal_width = st.slider("请输入花萼宽度:", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
        petal_length = st.slider("请输入花瓣长度:", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
        petal_width = st.slider("请输入花瓣宽度:", min_value=0.0, max_value=10.0, value=0.2, step=0.1)
        submit_button = st.form_submit_button(label='提交')
    feature_value = np.array([(sepal_length, sepal_width, petal_length, petal_width)])
    feature_prdt = clf.predict(feature_value)
    emoji = ":blue_heart:" if feature_prdt == 'Setosa' else (
        ":green_heart:" if feature_prdt == 'Virginica' else ":heart:")
    '你选择的算法为:', classifier
    '准确率:', round(acc,4)*100,'%' 
    st.write(f'预测结果为 -> *{feature_prdt[0]}* {emoji}')
