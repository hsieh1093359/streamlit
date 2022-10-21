import time

import streamlit as st

import numpy as np
import pandas as pd

number = 4
st.title('我的第一個應用程式')
st.title(number)

#表格
st.write("嘗試創建**表格**：")
df = pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
})
df

#圖表
chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['a', 'b', 'c'])
st.line_chart(chart_data)

#地圖
map_data = pd.DataFrame(
    np.random.randn(100, 2) / [50, 50] + [22.7, 120.3],
    columns=['lat', 'lon'])
st.map(map_data)
