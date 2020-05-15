import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
from  keras.preprocessing.text import text_to_word_sequence


#text
st.title("My First App")
st.header("My first header")
st.subheader("My first sub header")

#Dataframes

df = pd.DataFrame(data = np.random.randn(100, 3), columns= ["A", "B", "C"])
st.dataframe(df.head(5))

st.table(df.head(5))

st.write(df.head(5))





#Plots

fig = px.scatter(df, x = "A", y = "B")
fig

st.write(fig)


st.line_chart(df)

#Maps
map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

st.map(map_data)




# Widgets:


#checkbox:
if st.checkbox("Show data?"):
    st.dataframe(df.head(5))


# selectbox:
var = st.sidebar.selectbox("Select the X - Axis", df.columns.tolist())
fig = px.scatter(df, x = var , y = "B")
fig



#number input
my_rows = st.sidebar.number_input("Select Rows", min_value = 1, max_value= df.shape[0])
st.dataframe(df.head(my_rows))




#slider
no_rows = st.sidebar.slider("Select Rows", min_value = 1, max_value= df.shape[0])
st.dataframe(df.head(no_rows))






#text area
txt = st.text_area("Put in your txt", " We re currently learnign streamlit, and it will be so helpful")
st.write(text_to_word_sequence(txt))

#code
with st.echo():
    st.write("import streamlit as st")


#Latex

st.latex(r" A+ B * \sum_{k=0}^{i=1}")

#sidebar


#Media:
st.video("https://www.youtube.com/watch?v=8Ck8CpwMwGQ")
