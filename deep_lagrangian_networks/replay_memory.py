"""
经验回放缓冲区（Replay Memory）

该模块提供强化学习与动力学学习训练中常用的「经验池」数据结构，
用于暂存转移样本 (transitions)，并按小批量 (minibatch) 提供给优化器。

核心思想：
- 环境交互产生的样本往往时间上高度相关；随机或小批量打乱后再训练，可减弱相关性、稳定梯度。
- 固定容量的环形缓冲区在写满后覆盖最旧数据，在内存受限时近似保留近期经验。

存储方式：
- 基类 `ReplayMemory` 用若干块 NumPy 数组并列存储多路数据（例如状态、下一状态、动作等），
  形状为 (max_samples, *每路维度)。
- `PyTorchReplayMemory` 将存储改为 `torch.Tensor`，并可选放入 GPU。

注意：
- 文件末尾 `RandomBuffer` / `RandomReplayMemory` 依赖 `_x`、`_y` 字段，与当前 `ReplayMemory`
  使用的 `_data` 列表不一致，通常视为未完成或需自行对接基类的扩展草稿。
"""

import warnings  # Python 标准库：发出与捕获警告
import numpy as np  # NumPy 数值计算库
import numpy.random as random  # NumPy 随机数（与 np.random 等价接口的一部分）
import torch  # PyTorch 张量与 GPU 存储


def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    """
    将 warnings 格式化为单行字符串

    功能：
    - 默认的警告信息可能多行、不易和训练日志对齐；这里固定为「文件名:行号: 类别: 消息」。

    参数:
        message: 警告文本
        category: 警告类别类对象
        filename, lineno: 触发位置
        file, line: 可选，标准 warnings 钩子接口约定

    返回:
        str: 一行格式化后的警告字符串
    """
    return '%s:%s: %s: %s\n' % (filename, lineno, category.__name__, message)


# Hook：此后 warnings.warn 会使用上面的格式化函数
warnings.formatwarning = warning_on_one_line


class ReplayMemory(object):
    """
    环形经验池（NumPy 版）

    功能：
    - 按最大容量 `_max_samples` 循环写入；`_data_n` 记录当前有效样本条数（未满时递增，满后恒为容量上限）。
    - 支持 `for batch in memory:`：先将 0.._data_n-1 打乱，再每次取出 `_minibatch_size` 条；
      若剩余不足一整批则丢弃（不返回不满批）。

    参数说明（初始化）：
    - dim：list/tuple，每个元素是一路数据的形状（不含 batch），例如 [(obs_dim,), (act_dim,)]。
    """

    def __init__(self, maximum_number_of_samples, minibatch_size, dim):
        """
        初始化环形缓冲区

        参数:
            maximum_number_of_samples (int): 最大样本条数
            minibatch_size (int): 每次迭代返回的 batch 大小
            dim (list): 每路数据的形状元组列表，用于分配 self._data[i]
        """
        # General parameters（与论文/训练脚本中的容量、batch 尺寸一致）
        self._max_samples = maximum_number_of_samples  # 环形数组长度
        self._minibatch_size = minibatch_size  # 每个 minibatch 的样本数
        self._dim = dim  # 各路特征形状

        # Ring buffer 写指针：下一个写入位置落在 [0, _max_samples)
        self._data_idx = 0
        # 当前视为「有效」的样本数量（不超过 _max_samples）
        self._data_n = 0

        # Sampling：迭代打乱序列与游标
        self._sampler_idx = 0  # 当前在 permutation 中的起始下标
        self._order = None  # np.random.permutation 的结果；None 表示尚未进入迭代

        # Data structure：为每一路分配 (max_samples, *dim[i])
        self._data = []
        for i in range(len(dim)):
            self._data.append(np.empty((self._max_samples, ) + dim[i]))

    def __iter__(self):
        """
        开始一轮遍历：对有效下标 0.._data_n-1 做随机排列

        返回:
            self：迭代器协议要求返回自身
        """
        # Shuffle data and reset counter:
        # np.random.permutation(n) 生成 0..n-1 的一个随机排列，用于无放回地按 batch 取样
        self._order = np.random.permutation(self._data_n)
        self._sampler_idx = 0
        return self

    def __next__(self):
        """
        取下一个 minibatch；不足 minibatch_size 的尾部直接结束迭代（抛出 StopIteration）

        返回:
            list[np.ndarray]: 与 self._data 路数相同，每一路 shape=(minibatch_size, ...)
        """
        if self._order is None or self._sampler_idx >= self._order.size:
            raise StopIteration()

        tmp = self._sampler_idx
        self._sampler_idx += self._minibatch_size
        # 不要超过 permutation 长度，否则 batch_idx 可能为空或长度不对
        self._sampler_idx = min(self._sampler_idx, self._order.size)

        batch_idx = self._order[tmp:self._sampler_idx]

        # Reject batches that have less samples:
        # 训练侧通常要求固定 batch 大小，因此最后不足一批的数据直接舍弃
        if batch_idx.size < self._minibatch_size:
            raise StopIteration()

        # 各路数组用同一组索引取出子批次，保证每条样本在各路上对齐
        out = [x[batch_idx] for x in self._data]
        return out

    def add_samples(self, data):
        """
        写入一批样本（可从任意起点环形覆盖）

        参数:
            data (list): 长度等于 len(self._data)，data[i] 的第一维为本次写入条数 batch_size
        """
        assert len(data) == len(self._data)

        # Add samples:
        # 从 _data_idx 起连续写入 batch_size 条；下标对 _max_samples 取模实现环形
        add_idx = self._data_idx + np.arange(data[0].shape[0])
        add_idx = np.mod(add_idx, self._max_samples)

        for i in range(len(data)):
            self._data[i][add_idx] = data[i][:]

        # Update index:
        # 写指针移到「最后一次写入位置的下一格」；有效条数增加直至封顶
        self._data_idx = np.mod(add_idx[-1] + 1, self._max_samples)
        self._data_n = min(self._data_n + data[0].shape[0], self._max_samples)

        # Clear excessive GPU Memory:
        # 若 data 中含 GPU Tensor，del 可尽早释放 Python 引用，有利于显存回收
        del data

    def shuffle(self):
        """
        手动设置打乱顺序（使用当前 _data_idx 作为排列长度，与 __iter__ 使用的 _data_n 不同）

        说明:
        - 若仅用于调试或特定采样逻辑，请注意 _data_idx 与 _data_n 在环形缓冲中的语义差别。
        """
        self._order = np.random.permutation(self._data_idx)
        self._sampler_idx = 0

    def get_full_mem(self):
        """
        取出当前全部有效样本（仅前 _data_n 行）

        返回:
            list[np.ndarray]: 与 self._data 结构一致，但第一维为 _data_n
        """
        out = [x[:self._data_n] for x in self._data]
        return out

    def not_empty(self):
        """
        缓冲区是否非空

        返回:
            bool: _data_n > 0 时为 True
        """
        return self._data_n > 0


class PyTorchReplayMemory(ReplayMemory):
    """
    PyTorch 版环形经验池

    功能：
    - 与 ReplayMemory 相同的环形逻辑，但内部数组为 torch.Tensor。
    - cuda=True 时分配到 GPU，便于与神经网络同一设备上训练。
    """

    def __init__(self, max_samples, minibatch_size, dim, cuda):
        """
        参数:
            max_samples (int): 最大样本数
            minibatch_size (int): batch 大小
            dim (list): 各路形状
            cuda (bool): 是否使用 .cuda() 放到默认 GPU
        """
        super(PyTorchReplayMemory, self).__init__(max_samples, minibatch_size, dim)

        self._cuda = cuda
        for i in range(len(dim)):
            # 预分配与 ReplayMemory 相同形状的张量
            self._data[i] = torch.empty((self._max_samples,) + dim[i])

            if self._cuda:
                self._data[i] = self._data[i].cuda()

    def add_samples(self, data):
        """
        写入样本：numpy 先转为 torch.float，再用 type_as 与缓冲区 dtype/device 对齐

        参数:
            data (list): 每路可为 np.ndarray 或 torch.Tensor
        """
        # Cast Input Data:
        tmp_data = []

        for i, x in enumerate(data):
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float()

            # type_as：对齐 dtype 与 device（CPU/GPU），避免就地写入时报错
            tmp_data.append(x.type_as(self._data[i]))

        super(PyTorchReplayMemory, self).add_samples(tmp_data)


class PyTorchTestMemory(PyTorchReplayMemory):
    """
    测试 / 评估用顺序遍历

    功能：
    - __iter__ 使用顺序索引 np.arange(_data_n)，不打乱。
    - __next__ 允许最后一个 batch 小于 minibatch_size（与训练用 ReplayMemory 不同）。
    """

    def __init__(self, max_samples, minibatch_size, dim, cuda):
        super(PyTorchTestMemory, self).__init__(max_samples, minibatch_size, dim, cuda)

    def __iter__(self):
        # Reset counter:
        # 按存储顺序 0,1,...,_data_n-1 依次切块
        self._order = np.arange(self._data_n)
        self._sampler_idx = 0
        return self

    def __next__(self):
        if self._order is None or self._sampler_idx >= self._order.size:
            raise StopIteration()

        tmp = self._sampler_idx
        self._sampler_idx += self._minibatch_size
        self._sampler_idx = min(self._sampler_idx, self._order.size)

        batch_idx = self._order[tmp:self._sampler_idx]
        out = [x[batch_idx] for x in self._data]
        return out


class RandomBuffer(ReplayMemory):
    """
    随机抽取 minibatch 并从缓冲区删除被抽中的样本（设计意图）

    功能（代码层面意图）：
    - get_mini_batch：随机选索引，拷贝出 x_batch、y_batch，再用 np.delete 压缩缓冲区。

    注意：
    - 实现使用 self._x、self._y；基类 ReplayMemory 仅初始化 self._data。直接使用本类需与基类字段对齐。
    """

    def __init__(self, max_samples, minibatch_size, dim_input, dim_output, enforce_max_batch_size=False):
        super(RandomBuffer, self).__init__(max_samples, minibatch_size, dim_input, dim_output)

        # Parameters:
        self._enforce_max_batch_size = enforce_max_batch_size

    def get_mini_batch(self):
        """
        随机取一批数据并从缓冲中移除对应条目

        返回:
            tuple: (x_batch, y_batch)，若不可用则为 (None, None)
        """
        if self._data_n == 0 or (self._enforce_max_batch_size and self._data_n < self._minibatch_size):
            return None, None

        # Draw Random Mini-Batch:
        # random.choice(n, size)：在 [0, n) 上均匀抽样（此处 size 不超过当前存量）
        idx = random.choice(self._data_n, min(self._minibatch_size, self._data_n))
        x_batch = np.array(self._x[idx], copy=True)
        y_batch = np.array(self._y[idx], copy=True)

        # Note Faster with indexing:
        # 原注释：仅用索引视图可能更快，但删除与压缩逻辑会更复杂，此处用 copy + delete 实现清晰版本。

        # Remove Samples from Buffer:
        # np.delete(arr, idx, axis=0) 沿样本维删掉选中行，再把剩余部分写回缓冲区前部
        after_removal_x = np.delete(self._x, idx, 0)
        after_removal_y = np.delete(self._y, idx, 0)
        self._data_n -= idx.size

        if self._data_n > 0:
            self._x[0:self._data_n] = after_removal_x[0:self._data_n]
            self._y[0:self._data_n] = after_removal_y[0:self._data_n]

        return x_batch, y_batch

    def __next__(self):
        raise RuntimeError

    def __iter__(self):
        raise RuntimeError


class RandomReplayMemory(ReplayMemory):
    """
    随机覆盖式写入 (x, y) 的缓冲区（设计意图）

    功能（代码层面意图）：
    - 优先顺序填满空槽；若一批样本超出剩余连续空间，则多余部分随机写入已有位置（replace=False）。

    注意：
    - 同样依赖 self._x、self._y，需与基类存储约定一致。
    """

    def __init__(self, max_samples, minibatch_size, dim_input, dim_output):
        super(RandomReplayMemory, self).__init__(max_samples, minibatch_size, dim_input, dim_output)

    def add_samples(self, x, y):
        """
        添加一批样本；断言要求本批条数 n_samples < _max_samples

        参数:
            x, y (np.ndarray): 第一维为样本数，且两者条数一致
        """
        n_samples = x.shape[0]
        assert n_samples < self._max_samples

        # Add Samples in sequential order:
        # 从当前 _data_n 起连续写入，直到顶满 max_samples
        add_idx = np.arange(self._data_n, min(self._data_n + n_samples, self._max_samples))

        self._x[add_idx] = x[:add_idx.size]
        self._y[add_idx] = y[:add_idx.size]

        self._data_n += add_idx.size
        assert self._data_n <= self._max_samples

        # Add samples in random order:
        # 若还有剩余样本，则在已有 [0, _data_n) 中无放回随机选位置覆盖
        random_add_idx = random.choice(self._data_n, n_samples - add_idx.size, replace=False)

        self._x[random_add_idx] = x[add_idx.size:]
        self._y[random_add_idx] = y[add_idx.size:]

    def get_mini_batch(self):
        raise RuntimeError

    def __next__(self):
        raise RuntimeError

    def __iter__(self):
        raise RuntimeError

