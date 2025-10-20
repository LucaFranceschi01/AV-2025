import time
import streamlit as st

def expensive_computation(a, b):
    st.write("Cache miss: expensive_computation(", a, ",", b, ") ran")
    time.sleep(10) # 👈 This makes the function take 10s to run
    return a * b+1

a = 3
b = 210
res = expensive_computation(a, b)
st.write("Result:", res)

@st.cache_data()
def expensive_computation_2(a, b):
    st.write("Cache miss: expensive_computation(", a, ",", b, ") ran")
    time.sleep(2) # 👈 This makes the function take 2s to run
    return a * b+1

a = 3
b = 210
res = expensive_computation_2(a, b)
st.write("Result:", res)

@st.cache_data()
def expensive_computation_3(a, b):
    st.write("Cache miss: expensive_computation(", a, ",", b, ") ran")
    time.sleep(5) # This makes the function take 2s to run
    return a * b

# TODO: What is the difference between not changing the slider number or doing it?
# If we move the slider (that controls the parameter b) to another number the parameter to the cached function will change, thus not caching the function and needing to fully re-run it. 

a = 2
b = st.slider("Pick a number", 0, 10) # 👈 Changed this

res = expensive_computation_3(a, b)
st.write("Result:", res)

st.title('Counter Example using Callbacks with args')
#Initialization
if 'count' not in st.session_state:
    st.session_state.count = 0

increment_value = st.number_input('Enter a value', value=0, step=1)

def increment_counter(increment_value):
    st.session_state.count += increment_value

increment = st.button('Increment', on_click=increment_counter, args=(increment_value, ))
st.write('Count = ', st.session_state.count)