# -*- coding:utf-8 -*-
import streamlit as st
from streamlit_option_menu import option_menu

def main():
    st.set_page_config(layout="wide")
    with st.sidebar:
        selected = option_menu("Main Menu", ['Home', 'Description', 'Data', 'EDA', 'STAT', 'ML'],
                icons=['house', 'card-checklist', 'card-checklist', 'bar-chart', 'clipboard-data', 'clipboard-data'],
                menu_icon="cast", default_index=0, orientation = 'vertical', key='main_option')

    if selected == 'Home':
        pass
    elif selected == 'Description':
        pass
    elif selected == 'Data':
        pass
    elif selected == 'EDA':
        pass
    elif selected == 'STAT':
        pass
    elif selected == 'ML':
        pass
    else:
        print('error..')


if __name__ == "__main__":
    main()