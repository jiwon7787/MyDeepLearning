# your_script.py

from memory_profiler import profile
import weakref
import numpy as np

# Variableクラスの定義
class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError("{} is not supported".format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

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
            # gys = [output.grad for output in f.outputs] # 弱参照にしたため親から子を直接参照できない
            gys = [output().grad for output in f.outputs] # 弱参照のデータにアクセスするには（）を加える
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

    def cleargrad(self):
        self.grad = None


# Functionクラスの定義
class Function:
    def __call__(self, *inputs):
        self.generation = max([x.generation for x in inputs])
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        #self.outputs = outputs # 循環参照を引き起こす（親から子、子から親を参照）
        self.outputs = [weakref.ref(output) for output in outputs]  # 親から子に対して、参照カウントを増やさずに参照（弱参照）
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


# Squareクラスの定義
class Square(Function):
    def forward(self, x):
        return x**2

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


def square(x):
    return Square()(x)


def add(x0, x1):
    return Add()(x0, x1)


# メモリ使用量を計測する関数
@profile
def run_memory_profiler():
    for i in range(10000):
        x = Variable(np.random.randn(10000))
        y = square(square(square(x)))


if __name__ == "__main__":
    run_memory_profiler()
