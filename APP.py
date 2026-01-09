import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from rdkit import Chem
from rdkit.Chem import AllChem
import pickle

# 你的morgan_fp类
class morgan_fp:
    def __init__(self, radius, length):
        self.radius = radius
        self.length = length
    def __call__(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # 返回全0指纹或报错
            return np.zeros(self.length, dtype=np.float32)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, self.length)
        npfp = np.array(list(fp.ToBitString())).astype('float32')
        return npfp

# 特征处理函数
def conv_data(data, fp):
    data = data.copy()
    data['fp'] = data['SMILES'].apply(fp)

    x_fp = np.array(list(data['fp']))
    x1 = data['EHOMO'].values.reshape(-1, 1)
    x2 = data['Egap'].values.reshape(-1, 1)
    x3 = data['Dipole moment'].values.reshape(-1, 1)
    x4 = data['Polarizability'].values.reshape(-1, 1)
    x5 = data.loc[:, 'Br':'Cl2'].values

    xx = np.concatenate([x_fp, x1, x2, x3, x4, x5], axis=1)
    fp_cols = [f'fp_{i}' for i in range(x_fp.shape[1])]
    other_cols = ['EHOMO', 'Egap', 'Dipole moment', 'Polarizability']
    br_to_cl2_cols = data.loc[:, 'Br':'Cl2'].columns.tolist()
    columns = fp_cols + other_cols + br_to_cl2_cols

    xx_df = pd.DataFrame(xx, columns=columns)
    return xx_df

# 加载模型
@st.cache_resource
def load_model():
    with open('CatBoost.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# 初始化指纹计算器
fp = morgan_fp(1, 1024)

# Streamlit界面
st.title('CatBoost模型在线预测')

st.markdown("### 输入化学分子及物理化学特征")

EHOMO = st.number_input('EHOMO', value=-0.349420, format="%.6f")
Egap = st.number_input('Egap', value=0.382150, format="%.6f")
Dipole_moment = st.number_input('Dipole moment', value=2.5010, format="%.4f")
Polarizability = st.number_input('Polarizability', value=38.57, format="%.2f")
SMILES = st.text_input('SMILES', value='CCO')

st.markdown("### 取值为0或1，代表是否存在以下基团（独热编码）")
Br = st.selectbox('Br', options=[0,1], index=1)
Br2 = st.selectbox('Br2', options=[0,1], index=0)
Cl = st.selectbox('Cl', options=[0,1], index=0)
Cl2 = st.selectbox('Cl2', options=[0,1], index=0)

if st.button('预测'):
    input_dict = {
        'EHOMO': [EHOMO],
        'Egap': [Egap],
        'Dipole moment': [Dipole_moment],
        'Polarizability': [Polarizability],
        'SMILES': [SMILES],
        'Br': [Br],
        'Br2': [Br2],
        'Cl': [Cl],
        'Cl2': [Cl2]
    }
    input_df = pd.DataFrame(input_dict)

    # 特征处理
    X = conv_data(input_df, fp)

    # 载入模型
    model = load_model()

    # 预测
    pred = model.predict(X)[0]

    st.success(f'预测结果 logk = {pred:.6f}')
