import os
import subprocess
import urllib.request
from dezero import cuda



# 変数をDOT言語に変換
def _dot_var(v, verbose=False): # v: Variableインスタンス
    dot_var = '{} [label="{}", color=orange, style=filled]\n' # dot言語のフォーマット（{}部分はformat()の引数で置き換える）

    # Variableインスタンスに名前がなければ空白、あれば名前をnameに返す
    name = '' if v.name is None else v.name
    if verbose and v.data is not None: # Variableインスタンスの形状と型を出力する設定がオンかつそのインスタンスが実際データを保持する場合
        if v.name is not None: # Variableインスタンスに名前があれば
            name += ': ' # 文字列追加
        name += str(v.shape) + ' ' + str(v.dtype) # 形状と型の文字列追加

    return dot_var.format(id(v), name) # VariableインスタンスのオブジェクトIDと編集したnameをフォーマットに代入して返す

# 関数をDOT言語に変換＋入力と出力、関数ノードを矢印で結ぶ
def _dot_func(f, ferbose=False): # f: 関数インスタンス
    dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n' # dot言語のフォーマット（{}部分はformat()の引数で置き換える）
    txt = dot_func.format(id(f), f.__class__.__name__) # 関数インスタンスのオブジェクトIDと関数のクラス名をフォーマットに代入しtxtに返す

    dot_edge = '{} -> {}\n' # ノードを矢印で結ぶフォーマット
    for x in f.inputs: # 関数の入力
        txt += dot_edge.format(id(x), id(f)) # 入力→関数をdot_edgeフォーマットに代入してtxtに追加
    for y in f.outputs: # 関数の出力
        txt += dot_edge.format(id(f), id(y())) # 関数→出力をdot_edgeフォーマットに代入してtxtに追加
    return txt # 関数のノード化＋入出力と関数を矢印で結ぶ文字列が返される

def get_dot_graph(output, verbose=True):
    txt = ''
    funcs = []
    seen_set = set()

    # 初見の関数インスタンスだけfuncsリストに追加する関数（複数の出力を持つ関数が複数回funcsリストに入れないため）
    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            # funcs.sort(key = lambda x: x.generation) #  逆伝播みたいに出力側から順番に辿る必要はない（繋がりがわかりさえすればいい）
            seen_set.add(f)

    add_func(output.creator) # 最初(一番出力側)の関数だけ手動でfuncsリストに代入（それ以降の関数はループ内のadd_funcsで自動的にリストに追加される）
    txt += _dot_var(output, verbose) # 最初(一番出力側)の出力だけ手動でDOT言語変換するための関数を呼ぶ（それ以降はループ内）

    while funcs:
        func = funcs.pop()
        txt += _dot_func(func) # リストから得た関数をDOT言語に変換しtxtに追加＋矢印設定もtxtに追加
        for x in func.inputs:
            txt += _dot_var(x, verbose) # 関数の入力もDOT言語に変換しtxtに追加

            if x.creator is not None:
                add_func(x.creator) # ここで同じ関数が複数回呼ばれる場合があるからそれをadd_funcで弾いて一回だけリストに入れる
    return 'digraph g {\n' + txt + '}'

def plot_dot_graph(output, verbose=True, to_file='graph.png'):
    dot_graph = get_dot_graph(output, verbose)

    # 1.dotデータをファイルに保存
    #ホームディレクトリパスと.dezeroディレクトリ（そのディレクトリがあると仮定された）パスを結合して、.dezeroのパス（単に文字列）を生成
    tmp_dir = os.path.join(os.path.expanduser('~'), '.dezero') #C:\Users\jiwon\.dezero (一時的な保存先)
    # tmp_dirが存在しない場合、新しくディレクトリを作成
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir) # C:\Users\jiwon\.dezeroに新しくディレクトリ作成
    graph_path = os.path.join(tmp_dir, 'tmp_graph.dot') # dezeroディレクトリ内ににdotファイルを作成するためのパスを生成

    with open(graph_path, 'w') as f: # そのパス上にファイルを作成し
        f.write(dot_graph) # dot_graphの内容をそのファイルに書き込む

    # 2. dotコマンドを呼ぶ
    extension = os.path.splitext(to_file)[1][1:] # ('graph', '.png')に分割後、１番目の要素(.png)における１番目以降の要素(png)を取得
    # コマンドフォーマットにdotファイルのパス, 変換後の拡張子,ファイル名（出力先パス）を代入
    # ファイルはカレントディレクトリ（実行ファイルと同じディレクトリ）に出力される（to_fileの代わりにフルパスを入れればそこに出力される）
    cmd = 'dot {} -T {} -o {}'.format(graph_path, extension, to_file)
    subprocess.run(cmd, shell=True) # ターミナルでコマンド実行

    # Jupyterに対応させる（pythonファイルで実行するなら上記まででいい）
    try:
        from IPython import display
        return display.Image(filename=to_file)
    except:
        pass

def reshape_sum_backward(gy, x_shape, axis, keepdims):
    """Reshape gradient appropriately for dezero.functions.sum's backward.

    Args:
        gy (dezero.Variable): Gradient variable from the output by backprop.
        x_shape (tuple): Shape used at sum function's forward.
        axis (None or int or tuple of ints): Axis used at sum function's
            forward.
        keepdims (bool): Keepdims used at sum function's forward.

    Returns:
        dezero.Variable: Gradient variable which is reshaped appropriately
    """
    ndim = len(x_shape)
    tupled_axis = axis
    if axis is None:
        tupled_axis = None
    elif not isinstance(axis, tuple):
        tupled_axis = (axis,)

    if not (ndim == 0 or tupled_axis is None or keepdims):
        actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]
        shape = list(gy.shape)
        for a in sorted(actual_axis):
            shape.insert(a, 1)
    else:
        shape = gy.shape

    gy = gy.reshape(shape)  # reshape
    return gy

def sum_to(x, shape):
    """Sum elements along axes to output an array of a given shape.

    Args:
        x (ndarray): Input array.
        shape:

    Returns:
        ndarray: Output array of the shape.
    """
    ndim = len(shape)
    lead = x.ndim - ndim
    lead_axis = tuple(range(lead))

    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    y = x.sum(lead_axis + axis, keepdims=True)
    if lead > 0:
        y = y.squeeze(lead_axis)
    return y

def show_progress(block_num, block_size, total_size):
    bar_template = "\r[{}] {:.2f}%"

    downloaded = block_num * block_size
    p = downloaded / total_size * 100
    i = int(downloaded / total_size * 30)
    if p >= 100.0: p = 100.0
    if i >= 30: i = 30
    bar = "#" * i + "." * (30 - i)
    print(bar_template.format(bar, p), end='')

cache_dir = os.path.join(os.path.expanduser('~'), '.dezero')

def get_file(url, file_name=None):
    """Download a file from the `url` if it is not in the cache.

    The file at the `url` is downloaded to the `~/.dezero`.

    Args:
        url (str): URL of the file.
        file_name (str): Name of the file. It `None` is specified the original
            file name is used.

    Returns:
        str: Absolute path to the saved file.
    """
    if file_name is None:
        file_name = url[url.rfind('/') + 1:]
    file_path = os.path.join(cache_dir, file_name)

    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    if os.path.exists(file_path):
        return file_path

    print("Downloading: " + file_name)
    try:
        urllib.request.urlretrieve(url, file_path, show_progress)
    except (Exception, KeyboardInterrupt) as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise
    print(" Done")

    return file_path

def pair(x):
    if isinstance(x, int):
        return (x, x)
    elif isinstance(x, tuple):
        assert len(x) == 2
        return x
    else:
        raise ValueError

def logsumexp(x, axis=1):
    xp = cuda.get_array_module(x)
    m = x.max(axis=axis, keepdims=True)
    y = x - m
    xp.exp(y, out=y)
    s = y.sum(axis=axis, keepdims=True)
    xp.log(s, out=s)
    m += s
    return m

def get_conv_outsize(input_size, kernel_size, stride, pad):
    return (input_size + pad * 2 - kernel_size) // stride + 1

def get_deconv_outsize(size, k, s, p):
    return s * (size - 1) + k - 2 * p

def pair(x):
    if isinstance(x, int):
        return (x, x)
    elif isinstance(x, tuple):
        assert len(x) == 2
        return x
    else:
        raise ValueError
