import streamlit as st 
import langchain_helper  

st.title("Resturant name generator")
cuisine = st.sidebar.selectbox("Pick a cusine", ("Indian", "Italian", "Mexican", "Arabic"))


if cuisine:
    response = langchain_helper.generate_restaurant_name_and_items(cuisine)
    st.header(response['restaurant_name'].strip())
    menu_items = response['menu_items'].strip().split(",")
    st.write("**Menu Items**")
    for item in menu_items:
        st.write("-", item)
