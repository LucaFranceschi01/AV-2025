import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff

st.title('This is an example of title')

#The subheader
st.subheader("This is an example of subheader")

#Writing a sentence
st.write("This an example of sentence")

#Include emoticons
st.write("This streamlit app adds *different formats* and icons is as :sunglasses:")
st.write("This streamlit app adds *different formats* and icons is as :snow_cloud:")

#Let's create a sidebar

st.sidebar.header("The header of the sidebar")
st.sidebar.write("*Hello*")

#Basics line chart, area chart and bar chart
chart_data = pd.DataFrame(
np.random.randn(20, 3),
columns=['a', 'b', 'c'])
st.write("This is line chart")
st.line_chart(chart_data)
st.write("This is the area chart")
st.area_chart(chart_data)
st.write("This is the bar chart")
st.bar_chart(chart_data)

#Example 1
arr = np.random.normal(1, 1, size=100)
fig, ax = plt.subplots()
ax.hist(arr, bins=20)
st.write("Example 1 of plot with Matplotlib")
st.pyplot(fig)

#Seaborn: Seaborn builds on top of a Matplotlib figure so you can display the charts in the same way
penguins = sns.load_dataset("penguins")
st.dataframe(penguins[["species", "flipper_length_mm"]].sample(6))

# Create Figure beforehand
fig = plt.figure(figsize=(9, 7))
sns.histplot(data=penguins, x="flipper_length_mm", hue="species", multiple="stack")
plt.title("Hello Penguins!")
st.write("Example of a plot with Seaborn library")
st.pyplot(fig)

#Step 10: Show a dataframe table in your app
st.dataframe(penguins[["species", "flipper_length_mm"]].sample(6))

#Creating a map Maps
#Let's create randomly a lattitude and longitud variables
df = pd.DataFrame(
np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
columns=['lat', 'lon']) #These columns are totally necessary
st.write("Example of a plot with a map")
st.map(df)

# Add histogram data
x1 = np.random.randn(200) - 2
x2 = np.random.randn(200)
x3 = np.random.randn(200) + 2

# Group data together
hist_data = [x1, x2, x3]

group_labels = ['Group 1', 'Group 2', 'Group 3']

# Create distplot with custom bin_size
fig = ff.create_distplot(
hist_data, group_labels, bin_size=[.1, .25, .5])

# Plot!
st.write("Example of a plot with Plotly")
st.plotly_chart(fig, use_container_width=True)