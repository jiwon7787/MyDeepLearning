# simple_core.pyをアプデ（逆伝播の計算グラフを取得し高階微分を可能にする（最適化のニュートン法を可能にするため））
import weakref
import numpy as np
import contextlib
import sys
sys.path.append(r'C:\Users\jiwon\OneDrive\Desktop\DL3')
import dezero


# =============================================================================
# Config
# =============================================================================
class Config:
    enable_backprop = True
    train = True


@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

def no_grad():
    return using_config('enable_backprop', False)

def test_mode():
    return using_config('train', False)


# =============================================================================
# Variable / Function
# =============================================================================

# CuPyもNumpyも両対応させる
try:
    import cupy
    array_types = (np.ndarray, cupy.ndarray)
except ImportError:
    array_types = (np.ndarray)


class Variable:
    __array_priority__ = 200

    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, array_types): # NumpyとCuPy両対応
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self):
        self.grad = None

    def backward(self, retain_grad=False, create_graph=False):
        if self.grad is None:
            # self.grad = np.ones_like(self.data)
            xp = dezero.cuda.get_array_module(self.data) # データからnpかcpを取得
            self.grad = Variable(xp.ones_like(self.data)) # 逆伝播の計算グラフも生成できるようにVariableインスタンスにする

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            # gys = [output.grad for output in f.outputs] # 親（関数）から子(変数)を直接参照する場合（循環参照を引き起こす）
            gys = [output().grad for output in f.outputs]  # 弱参照のデータにアクセスするには（）を加える

            # スイッチのデフォルト(Trueを想定)をFalseに切り替えることで、逆伝播計算（e.g. gy * x1）でまたFunctionインスタンスが呼ばれその計算に対する逆伝播（２次微分）をConfig.enable_backpropで止める）
            # 逆伝播が終わったらデフォルトに戻す（続きの逆伝播をするため）
            with using_config('enable_backprop', create_graph):
                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)

                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx

                    if x.creator is not None:
                        add_func(x.creator)

                if not retain_grad:
                    for y in f.outputs:
                        # y.grad = None # 親（関数）から子(変数)を直接参照する場合（循環参照を引き起こす）
                        y().grad = None  # 弱参照のデータにアクセスするには（）を加える

    def reshape(self, *shape):
        # 要素が一つのタプルかリスト(6,)や((2,3),)の場合、その要素shape[0]を取り出してreshape関数に代入
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return dezero.functions.reshape(self, shape) # F.reshapeを使うと循環インポートになるためここで直接呼ぶ

    # def transpose(self):
    #     return dezero.functions.transpose(self)

    def transpose(self, *axes):
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1:
            if isinstance(axes[0], (tuple, list)) or axes[0] is None:
                axes = axes[0]
        return dezero.functions.transpose(self, axes)

    @property
    def T(self):
        return dezero.functions.transpose(self) # x.T(インスタンス減数)でもx.transpose()でも同じことができる

    def sum(self, axis=None, keepdims=False):
        return dezero.functions.sum(self, axis, keepdims)

    def to_cpu(self):
        if self.data is not None:
            self.data = dezero.cuda.as_numpy(self.data)

    def to_gpu(self):
        if self.data is not None:
            self.data = dezero.cuda.as_cupy(self.data)

    def unchain(self):
        self.creator = None

    def unchain_backward(self):
        if self.creator is not None:
            funcs = [self.creator]
            while funcs:
                f = funcs.pop()
                for x in f.inputs:
                    if x.creator is not None:
                        funcs.append(x.creator)
                        x.unchain()

class Parameter(Variable):
    pass


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


# def as_array(x):
#     if np.isscalar(x):
#         return np.array(x)
#     return x

def as_array(x, array_module=np):
    if np.isscalar(x):
        return array_module.array(x)
    return x


class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]

        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            # self.outputs = outputs # 循環参照を引き起こす（親から子、子から親が参照し合っている）# Jupyterではこれでやる
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


# =============================================================================
# 四則演算 / 演算子のオーバーロード
# =============================================================================
# class Add(Function):
#     def forward(self, x0, x1):
#         y = x0 + x1
#         return y

#     def backward(self, gy):
#         return gy, gy

class Add(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape # 入力の形状を保持（逆伝播の時元の形状に戻すため）
        y = x0 + x1
        return y

    def backward(self, gy):
        gx0, gx1 = gy, gy
        # 前伝播計算時に形状が違っていたら（ブロードキャストによって形状が補正されていたら）
        if self.x0_shape != self.x1_shape:
            # 勾配の形状を入力の形状にしながらその合計を求める
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


# def add(x0, x1):
#     x1 = as_array(x1)
#     return Add()(x0, x1)

def add(x0, x1):
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Add()(x0, x1)


# class Mul(Function):
#     def forward(self, x0, x1):
#         y = x0 * x1
#         return y

#     def backward(self, gy):
#         # x0, x1 = self.inputs[0].data, self.inputs[1].data
#         x0, x1 = self.inputs # Variableインスタンスをそのまま計算に使う（順伝播みたいに）
#         return gy * x1, gy * x0 # Variableの特殊メソッドで演算子をそのまま使って計算できる

class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy * x1
        gx1 = gy * x0
        if x0.shape != x1.shape:  # for broadcast
            gx0 = dezero.functions.sum_to(gx0, x0.shape)
            gx1 = dezero.functions.sum_to(gx1, x1.shape)
        return gx0, gx1


# def mul(x0, x1):
#     x1 = as_array(x1)
#     return Mul()(x0, x1)

def mul(x0, x1):
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Mul()(x0, x1)


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


def neg(x):
    return Neg()(x)


# class Sub(Function):
#     def forward(self, x0, x1):
#         y = x0 - x1
#         return y

#     def backward(self, gy):
#         return gy, -gy

class Sub(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 - x1
        return y

    def backward(self, gy):
        gx0 = gy
        gx1 = -gy
        if self.x0_shape != self.x1_shape:  # for broadcast
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1

# def sub(x0, x1):
#     x1 = as_array(x1)
#     return Sub()(x0, x1)


# def rsub(x0, x1):
#     x1 = as_array(x1)
#     return Sub()(x1, x0)

def sub(x0, x1):
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Sub()(x1, x0)


# class Div(Function):
#     def forward(self, x0, x1):
#         y = x0 / x1
#         return y

#     def backward(self, gy):
#         # x0, x1 = self.inputs[0].data, self.inputs[1].data
#         x0, x1 = self.inputs
#         gx0 = gy / x1
#         gx1 = gy * (-x0 / x1 ** 2)
#         return gx0, gx1

class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        if x0.shape != x1.shape:  # for broadcast
            gx0 = dezero.functions.sum_to(gx0, x0.shape)
            gx1 = dezero.functions.sum_to(gx1, x1.shape)
        return gx0, gx1

# def div(x0, x1):
#     x1 = as_array(x1)
#     return Div()(x0, x1)


# def rdiv(x0, x1):
#     x1 = as_array(x1)
#     return Div()(x1, x0)

def div(x0, x1):
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Div()(x1, x0)

class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y

    def backward(self, gy):
        # x = self.inputs[0].data
        x, = self.inputs
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx



def pow(x, c):
    return Pow(c)(x)

# 演算子のオーバーロード（Variableクラスに特殊メソッドの実装＝演算子を直接使えるようにする）
def setup_variable():
    # Variableクラスに直接書かず関数でまとめてVariableクラスに追加する
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow
    Variable.__getitem__ = dezero.functions.get_item

