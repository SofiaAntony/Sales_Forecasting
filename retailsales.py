
import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
from PIL import Image
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import streamlit.components.v1 as components


_model = joblib.load('sales_.pkl')



st.set_page_config(page_title="Retails Sales Prediction", page_icon=None, layout="wide")

st.header("Retail Sales Prediction")
t1,t2,t3,t4,t5,t6,t7,t8= st.tabs(["Home","Retail Sales Prediction","EDA","Data Visualisation","Store based Insight","Department based insight","Insights on Holiday and Markdowns","Recommendations"])

with t1:
      with st.container():
         
          st.markdown("In the dynamic landscape of retail, accurate sales predictions are crucial for optimizing inventory, managing resources efficiently, and enhancing overall business performance. Our Retail Sales Prediction Project leverages cutting-edge data analytics and machine learning techniques to forecast sales trends, providing valuable insights for strategic decision-making.")
          i = Image.open('sa.png')
          st.image(i,use_column_width='always')
with t4:
    imag = Image.open('sal.png')
    st.image(imag,use_column_width='always')

with t8:
    ima = Image.open('ret.png')
    st.image(ima)
with t7:
    t = """
               <div class='tableauPlaceholder' id='viz1701025341402' style='position: relative'>
            <noscript>
                <a href='#'>
                    <img alt='Dashboard 2' src='https://public.tableau.com/static/images/av/avg_dep/Dashboard2/1_rss.png' style='border: none' />
                </a>
            </noscript>
            <object class='tableauViz' style='display:none;'>
                <param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' />
                <param name='embed_code_version' value='3' />
                <param name='site_root' value='' />
                <param name='name' value='avg_dep/Dashboard2' />
                <param name='tabs' value='no' />
                <param name='toolbar' value='yes' />
                <param name='static_image' value='https://public.tableau.com/static/images/av/avg_dep/Dashboard2/1.png' />
                <param name='animate_transition' value='yes' />
                <param name='display_static_image' value='yes' />
                <param name='display_spinner' value='yes' />
                <param name='display_overlay' value='yes' />
                <param name='display_count' value='yes' />
                <param name='language' value='en-US' />
            </object>
        </div>
        <script type='text/javascript'>
            var divElement = document.getElementById('viz1701025341402');
            var vizElement = divElement.getElementsByTagName('object')[0];
            if (divElement.offsetWidth > 800) {
                vizElement.style.width='100%';
                vizElement.style.height=(divElement.offsetWidth*0.75)+'px';
            } else if (divElement.offsetWidth > 500) {
                vizElement.style.width='100%';
                vizElement.style.height=(divElement.offsetWidth*0.75)+'px';
            } else {
                vizElement.style.width='100%';
                vizElement.style.height='777px';
            }
            var scriptElement = document.createElement('script');
            scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
            vizElement.parentNode.insertBefore(scriptElement, vizElement);
        </script>
        """
        
    st.components.v1.html(t, height = 4000, scrolling = True)
    

with t6:
    tabcode1 = """
        <div class='tableauPlaceholder' id='viz1701024908517' style='position: relative'>
            <noscript>
                <a href='#'>
                    <img alt='Dashboard 4' src='https://public.tableau.com/static/images/ko/ko_17008668991540/Dashboard4/1_rss.png' style='border: none' />
                </a>
            </noscript>
            <object class='tableauViz' style='display:none;'>
                <param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' />
                <param name='embed_code_version' value='3' />
                <param name='site_root' value='' />
                <param name='name' value='ko_17008668991540/Dashboard4' />
                <param name='tabs' value='no' />
                <param name='toolbar' value='yes' />
                <param name='static_image' value='https://public.tableau.com/static/images/ko/ko_17008668991540/Dashboard4/1.png' />
                <param name='animate_transition' value='yes' />
                <param name='display_static_image' value='yes' />
                <param name='display_spinner' value='yes' />
                <param name='display_overlay' value='yes' />
                <param name='display_count' value='yes' />
                <param name='language' value='en-US' />
            </object>
        </div>
        <script type='text/javascript'>
            var divElement = document.getElementById('viz1701024908517');
            var vizElement = divElement.getElementsByTagName('object')[0];
            if (divElement.offsetWidth > 800) {
                vizElement.style.width='1000px';
                vizElement.style.height='827px';
            } else if (divElement.offsetWidth > 500) {
                vizElement.style.width='1000px';
                vizElement.style.height='827px';
            } else {
                vizElement.style.width='100%';
                vizElement.style.height='1327px';
            }
            var scriptElement = document.createElement('script');
            scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
            vizElement.parentNode.insertBefore(scriptElement, vizElement);
        </script>
        """
    st.components.v1.html(tabcode1, height = 4000, scrolling = True)

with t5:
    
    tabcode = """
            <div class='tableauPlaceholder' id='viz1701022136935' style='position: relative'>
          <noscript>
            <a href='#'>
              <img alt='Dashboard 3 ' src='https://public.tableau.com/static/images/st/storess__/Dashboard3/1_rss.png' style='border: none' />
            </a>
          </noscript>
          <object class='tableauViz' style='display:none;'>
            <param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' />
            <param name='embed_code_version' value='3' />
            <param name='site_root' value='' />
            <param name='name' value='storess__&#47;Dashboard3' />
            <param name='tabs' value='no' />
            <param name='toolbar' value='yes' />
            <param name='static_image' value='https://public.tableau.com/static/images/st/storess__/Dashboard3/1.png' />
            <param name='animate_transition' value='yes' />
            <param name='display_static_image' value='yes' />
            <param name='display_spinner' value='yes' />
            <param name='display_overlay' value='yes' />
            <param name='display_count' value='yes' />
            <param name='language' value='en-US' />
          </object>
        </div>
        <script type='text/javascript'>
          var divElement = document.getElementById('viz1701022136935');
          var vizElement = divElement.getElementsByTagName('object')[0];
          if (divElement.offsetWidth > 800) {
            vizElement.style.width='1100px';
            vizElement.style.height='827px';
          } else if (divElement.offsetWidth > 500) {
            vizElement.style.width='1100px';
            vizElement.style.height='827px';
          } else {
            vizElement.style.width='100%';
            vizElement.style.height='1227px';
          }
          var scriptElement = document.createElement('script');
          scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
          vizElement.parentNode.insertBefore(scriptElement, vizElement);
        </script>
        """

    st.components.v1.html(tabcode, height = 4000, scrolling = True)

with t3:
    image = Image.open('sales.png')
    st.image(image)

with t2:
    col1,col2 = st.columns(2)
    with col1:
        year = st.text_input('Enter the Year of sales')
        type_ = st.selectbox('Enter the Sales Type(A:1,B:0,C:-1)',(-1,0,1))
        dept = st.selectbox('Enter the Department code No',( 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 16, 17, 18,
                19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
               36, 37, 38, 40, 41, 42, 44, 45, 46, 47, 48, 49, 51, 52, 54, 55, 56,
               58, 59, 60, 67, 71, 72, 74, 79, 80, 81, 82, 83, 85, 87, 90, 91, 92,
               93, 94, 95, 97, 98, 78, 96, 99, 77, 39, 50, 43, 65))
        cpi = st.text_input('CPI')
        fuel = st.text_input("Estimated Fuel Prize")
        m1 = st.text_input('Markdown Effect 1')
        m3 =st.text_input('Markdown Effect 3')
        m5 =st.text_input('Markdown Effect 5')
        size = st.text_input('Area size of the Shop {Generally 30K to 250K}')


    with col2:
        month = st.text_input('Enter the Month of sales')
        day = st.text_input('Enter the Day of the month')
        store = st.selectbox('Enter the Store No',( 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
               18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
               35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45))
        temp = st.text_input('Temperature')
        unemp = st.text_input('UnEmployment Rate {Generally ranges between 1 to 20}')
        holiday = st.selectbox('Is it a Holiday day {True:1,False:0} ',(0,1))
        m2 = st.text_input('Markdown Effect 2')
        m4 =st.text_input('Markdown Effect 4')
        

    c1,c2,c3 = st.columns([3,1,3])
    with c2:
        submit = st.button("Predict")
        if submit:
            data = np.array([[store, temp, fuel, m1, m2, m3, m4, m5, cpi, unemp, holiday, dept, type_, size, year, month, day]])
            y_pred = _model.predict(data)
            # Assuming y_pred is a numeric value
            st.success("The Weekly sales are {}".format(y_pred))

            
               




