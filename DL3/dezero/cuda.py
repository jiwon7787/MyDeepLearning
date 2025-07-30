# CupyもNumpyも対応させるためのファイル

# Cupyがインストールされていなくてもエラーを出さず代わりにNumpyを使えるようにする
import numpy as np
gpu_enable = True
try:
    import cupy as cp
    cupy = cp
except ImportError:
    gpu_enable = False
import sys
sys.path.append(r'C:\Users\jiwon\OneDrive\Desktop\DL3')
from dezero import Variable

# Variableにも対応したCupyとNumpy両対応のメソッドを定義

# 対応したモジュールを返す
def get_array_module(x):
    if isinstance(x, Variable):
        x = x.data

    if not gpu_enable: # インポート時にCupyがインストールされていないならNumpyモジュールを返す
        return np
    xp = cp.get_array_module(x)
    return xp

# numpy配列に変換する
def as_numpy(x):
    if isinstance(x, Variable):
        x = x.data

    if np.isscalar(x): # numpyスカラをnumpy配列に
        return np.array(x)
    elif isinstance(x, np.ndarray): # numpy配列はそのまま
        return x
    return cp.asnumpy(x) # Cupy配列はnumpy配列に

# Cupy配列に変換する
def as_cupy(x):
    if isinstance(x, Variable):
        x = x.data

    if not gpu_enable:
        raise Exception('Cupy cannot be loaded. Install CuPy!')
    return cp.asarray(x)
