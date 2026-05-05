"""
深度拉格朗日网络（DeepLagrangianNetwork）实现

该模块实现了基于拉格朗日力学的神经网络架构，用于学习机器人动力学模型。

核心思想：
- 使用神经网络学习系统的势能V(q)和动能T(q, q̇)
- 通过拉格朗日方程计算力矩：τ = d/dt(∂L/∂q̇) - ∂L/∂q
- 其中拉格朗日量 L = T - V（动能 - 势能）

物理意义：
- 力矩分解为：τ = H(q)q̈ + c(q,q̇) + g(q)
  - H(q): 惯性矩阵（来自动能）
  - c(q,q̇): 科里奥利力和离心力
  - g(q): 重力（来自势能）

网络结构保证：
- 惯性矩阵H正定对称（通过Cholesky分解H = L×L^T实现）
- 能量守恒（通过欧拉-拉格朗日方程约束）
"""

import torch  # PyTorch深度学习框架
import torch.nn as nn  # PyTorch神经网络模块
import torch.nn.functional as F  # PyTorch函数式接口
import numpy as np  # NumPy数值计算库


class LowTri:
    """
    下三角矩阵辅助类
    
    功能：
    - 将一维向量转换为下三角矩阵
    - 用于构造惯性矩阵H的Cholesky分解中的L矩阵
    - 确保H = L×L^T 正定对称
    
    数学原理：
    - 任何正定对称矩阵都可以分解为H = L×L^T
    - 其中L是下三角矩阵（对角线及以下非零）
    - 我们只学习L的下三角元素，减少参数数量
    """

    def __init__(self, m):
        """
        初始化下三角矩阵生成器
        
        参数:
            m (int): 矩阵的维度（m×m方阵）
        """
        # Calculate lower triangular matrix indices using numpy
        # 使用NumPy计算下三角矩阵的索引位置
        # np.tril_indices(m) 返回两个数组：
        #   - 第一个数组：行索引
        #   - 第二个数组：列索引
        # 例如：m=3时，返回 (array([0,1,1,2,2,2]), array([0,0,1,0,1,2]))
        # 对应位置：(0,0), (1,0), (1,1), (2,0), (2,1), (2,2)
        self._m = m  # 矩阵维度
        self._idx = np.tril_indices(self._m)  # 下三角元素的索引

    def __call__(self, l):
        """
        将一维向量转换为下三角矩阵
        
        参数:
            l (torch.Tensor): 包含下三角元素的一维向量
                             长度为 m×(m+1)/2
        
        返回:
            torch.Tensor: 下三角矩阵，shape=(batch_size, m, m)
        """
        # 获取批次大小（batch size）
        # batch_size: 同时处理多少个样本
        batch_size = l.shape[0]
        
        # 初始化下三角矩阵（全零）
        # shape=(batch_size, m, m)，类型与输入l相同
        self._L = torch.zeros(batch_size, self._m, self._m).type_as(l)

        # Assign values to matrix:
        # 将向量l中的值填入下三角矩阵的对应位置
        # self._idx[0]: 行索引数组
        # self._idx[1]: 列索引数组
        # l[:]: 所有元素
        # 这行代码实现了：对每个样本，将l的元素按索引填入L的下三角部分
        self._L[:batch_size, self._idx[0], self._idx[1]] = l[:]
        return self._L[:batch_size]


class SoftplusDer(nn.Module):
    """
    Softplus激活函数的导数
    
    Softplus函数：f(x) = ln(1 + e^x)
    导数：f'(x) = e^x / (1 + e^x) = 1 / (1 + e^(-x)) = sigmoid(βx)
    
    注意：这是Softplus的导数，不是Softplus本身！
    用于计算神经网络输出的梯度（用于物理量的导数）
    """
    
    def __init__(self, beta=1.):
        """
        初始化Softplus导数
        
        参数:
            beta (float): 缩放参数，控制函数的陡峭程度
        """
        super(SoftplusDer, self).__init__()  # 调用父类nn.Module的初始化
        self._beta = beta  # 存储缩放参数

    def forward(self, x):
        """
        前向传播：计算Softplus的导数
        
        参数:
            x (torch.Tensor): 输入张量
        
        返回:
            torch.Tensor: Softplus的导数值（即sigmoid函数）
        """
        # 数值稳定性处理：将x限制在[-20, 20]范围内
        # 防止exp(x)过大或过小导致数值溢出
        cx = torch.clamp(x, -20., 20.)
        
        # 计算指数函数：e^(β×x)
        exp_x = torch.exp(self._beta * cx)
        
        # Softplus的导数 = sigmoid(β×x) = e^(βx) / (1 + e^(βx))
        out = exp_x / (exp_x + 1.0)

        # 检查是否有NaN（非数字）值
        # NaN通常由数值不稳定导致（如0/0、∞-∞等）
        if torch.isnan(out).any():
            print("SoftPlus Forward output is NaN.")
        return out


class ReLUDer(nn.Module):
    """
    ReLU激活函数的导数
    
    ReLU函数：f(x) = max(0, x)
    导数：f'(x) = 1 if x > 0, 0 if x < 0
          （在x=0处导数未定义，通常取0或0.5）
    
    这里用ceil(clamp(x, 0, 1))实现：
    - clamp(x, 0, 1): 将x限制在[0,1]
      - 如果x>0，保持x值
      - 如果x≤0，变成0
    - ceil(...): 向上取整
      - 如果0 < x < 1，变成1
      - 如果x=0，还是0
    """
    
    def __init__(self):
        """初始化ReLU导数"""
        super(ReLUDer, self).__init__()

    def forward(self, x):
        """
        前向传播：计算ReLU的导数
        
        参数:
            x (torch.Tensor): 输入张量（ReLU之前的激活值）
        
        返回:
            torch.Tensor: ReLU的导数值（0或1）
        """
        # torch.clamp(x, 0, 1): 将x限制在[0, 1]范围
        #   x>0 → 保持不变
        #   x≤0 → 变成0
        # torch.ceil(...): 向上取整
        #   0 < x < 1 → 变成1
        #   x=0 → 还是0
        # 因此结果：x>0时为1，x≤0时为0（ReLU的导数）
        return torch.ceil(torch.clamp(x, 0, 1))


class Linear(nn.Module):
    """
    线性激活函数（恒等映射）
    
    f(x) = x
    
    用于不需要非线性的层（如输出层）
    """
    
    def __init__(self):
        """初始化线性激活函数"""
        super(Linear, self).__init__()

    def forward(self, x):
        """
        前向传播：返回输入本身
        
        参数:
            x (torch.Tensor): 输入张量
        
        返回:
            torch.Tensor: 输入张量（不变）
        """
        return x


class LinearDer(nn.Module):
    """
    线性函数的导数
    
    f(x) = x 的导数是 f'(x) = 1
    
    用于计算线性层的梯度
    """
    
    def __init__(self):
        """初始化线性导数"""
        super(LinearDer, self).__init__()

    def forward(self, x):
        """
        前向传播：返回全1张量
        
        参数:
            x (torch.Tensor): 输入张量（形状决定输出形状）
        
        返回:
            torch.Tensor: 全1张量（与输入形状相同）
        """
        # torch.clamp(x, 1, 1): 强制所有元素为1
        # 无论输入是什么，输出都是1（线性函数的导数恒为1）
        return torch.clamp(x, 1, 1)


class Cos(nn.Module):
    """
    余弦激活函数
    
    f(x) = cos(x)
    
    用于周期性问题的建模
    """
    
    def __init__(self):
        """初始化余弦激活函数"""
        super(Cos, self).__init__()

    def forward(self, x):
        """
        前向传播：计算余弦值
        
        参数:
            x (torch.Tensor): 输入张量（角度，单位：弧度）
        
        返回:
            torch.Tensor: 余弦值
        """
        return torch.cos(x)


class CosDer(nn.Module):
    """
    余弦函数的导数
    
    cos(x) 的导数是 -sin(x)
    
    用于计算余弦层的梯度
    """
    
    def __init__(self):
        """初始化余弦导数"""
        super(CosDer, self).__init__()

    def forward(self, x):
        """
        前向传播：计算余弦的导数
        
        参数:
            x (torch.Tensor): 输入张量
        
        返回:
            torch.Tensor: -sin(x)
        """
        # cos(x)的导数是 -sin(x)
        return -torch.sin(x)


class LagrangianLayer(nn.Module):
    """
    拉格朗日网络层
    
    这是DeLaN的核心层，实现了：
    1. 前向传播：计算层的输出
    2. 梯度传播：计算输出对输入的导数（雅可比矩阵）
    
    关键创新：
    - 不仅计算输出，还计算输出的梯度
    - 使用链式法则自动计算雅可比矩阵
    - 物理量的导数通过反向传播自动获得
    
    数学原理：
    如果 y = g(a)，其中 a = Wx + b
    那么 ∂y/∂x = g'(a) × W
    
    这里：
    - out = g(a): 层的输出（经过激活函数）
    - der = ∂y/∂x: 输出对输入的导数
    """

    def __init__(self, input_size, n_dof, activation="ReLu"):
        """
        初始化拉格朗日层
        
        参数:
            input_size (int): 输入特征的维度
            n_dof (int): 自由度数（输出维度）
            activation (str): 激活函数类型
                - "ReLu": 修正线性单元
                - "SoftPlus": 平滑的ReLU变体
                - "Cos": 余弦函数
                - "Linear": 线性函数
        """
        super(LagrangianLayer, self).__init__()

        # Create layer weights and biases:
        # 创建层的权重和偏置
        self.n_dof = n_dof  # 自由度数
        self.weight = nn.Parameter(torch.Tensor(n_dof, input_size))  # 权重矩阵：n_dof × input_size
        self.bias = nn.Parameter(torch.Tensor(n_dof))  # 偏置向量：n_dof

        # Initialize activation function and its derivative:
        # 初始化激活函数及其导数
        # 激活函数g：用于非线性变换
        # 导数g'：用于计算梯度
        if activation == "ReLu":
            self.g = nn.ReLU()  # ReLU激活函数
            self.g_prime = ReLUDer()  # ReLU的导数

        elif activation == "SoftPlus":
            self.softplus_beta = 1.0  # Softplus的缩放参数
            self.g = nn.Softplus(beta=self.softplus_beta)  # Softplus激活函数
            self.g_prime = SoftplusDer(beta=self.softplus_beta)  # Softplus的导数

        elif activation == "Cos":
            self.g = Cos()  # 余弦激活函数
            self.g_prime = CosDer()  # 余弦的导数

        elif activation == "Linear":
            self.g = Linear()  # 线性激活函数（恒等映射）
            self.g_prime = LinearDer()  # 线性函数的导数（恒为1）

        else:
            # 如果激活函数类型不支持，抛出错误
            raise ValueError("Activation Type must be in ['Linear', 'ReLu', 'SoftPlus', 'Cos'] but is {0}".format(self.activation))

    def forward(self, q, der_prev):
        """
        前向传播：计算层输出和梯度
        
        参数:
            q (torch.Tensor): 输入张量，shape=(batch_size, input_size)
            der_prev (torch.Tensor): 上一层的梯度，shape=(batch_size, prev_output, input)
        
        返回:
            out (torch.Tensor): 层输出，shape=(batch_size, n_dof)
            der (torch.Tensor): 输出对输入的导数，shape=(batch_size, n_dof, input_size)
        
        数学原理：
        1. 仿射变换：a = Wx + b
        2. 激活：out = g(a)
        3. 梯度（链式法则）：∂out/∂q = g'(a) × W × ∂prev_out/∂q
        """
        # Apply Affine Transformation:
        # 应用仿射变换（线性层）：a = W×q + b
        # F.linear: 矩阵乘法函数，等价于 torch.matmul(q, W.T) + b
        # 这里W是n_dof×input_size，q是batch_size×input_size
        # 输出a的shape是batch_size×n_dof
        a = F.linear(q, self.weight, self.bias)
        
        # 应用激活函数：out = g(a)
        out = self.g(a)
        
        # Compute gradient using chain rule:
        # 使用链式法则计算导数（雅可比矩阵）
        # 数学原理：∂out/∂q = ∂g/∂a × ∂a/∂q × ∂prev/∂q
        #                   = g'(a) × W × der_prev
        
        # 1. 计算激活函数的导数：g'(a)
        # g_prime(a)的shape是(batch_size, n_dof)
        # .view(-1, n_dof, 1): 变形为(batch_size, n_dof, 1)以便广播
        # * self.weight: 广播乘法，变成(batch_size, n_dof, input_size)
        #   g'(a)的第i行（n_dof个元素）乘以W的第i行（input_size个元素）
        g_prime_a = self.g_prime(a).view(-1, self.n_dof, 1) * self.weight
        
        # 2. 矩阵乘法：g'(a)×W × der_prev
        # g_prime_a: (batch_size, n_dof, input_size)
        # der_prev: (batch_size, input_size, input_size) 或类似
        # matmul: 矩阵乘法，最后两个维度相乘
        # 输出der: (batch_size, n_dof, input_size)
        der = torch.matmul(g_prime_a, der_prev)
        
        return out, der


class DeepLagrangianNetwork(nn.Module):
    """
    深度拉格朗日网络（DeepLagrangianNetwork）
    
    这是整个模型的核心类，实现了基于拉格朗日力学的动力学模型。
    
    网络架构：
    1. 共享骨干网络：提取位置特征
    2. 三个输出头：
       - 网络g：输出势能V(q)，求导得到重力g(q)
       - 网络lo：输出L的下三角非对角元素
       - 网络ld：输出L的对角元素（必须为正，用ReLU保证）
    3. 组合L矩阵，计算H = L×L^T（正定对称的惯性矩阵）
    4. 计算各项物理量：
       - 惯性力：H(q) × q̈
       - 科里奥利力：dH/dt × q̇ - 0.5 × q̇^T × dH/dq × q̇
       - 重力：∂V/∂q
    
    物理约束保证：
    - H正定对称：通过Cholesky分解H = L×L^T
    - 能量守恒：通过欧拉-拉格朗日方程
    """

    def __init__(self, n_dof, **kwargs):
        """
        初始化深度拉格朗日网络
        
        参数:
            n_dof (int): 自由度数（机械臂关节数）
            **kwargs: 超参数字典
                - n_width: 隐藏层宽度（默认128）
                - n_depth: 隐藏层深度（默认1）
                - b_init: 偏置初始化值（默认0.1）
                - b_diag_init: 对角元素偏置初始化（默认0.1）
                - w_init: 权重初始化方法（'xavier_normal'）
                - g_hidden: 隐藏层增益（默认√2）
                - g_output: 输出层增益（默认0.125）
                - p_sparse: 稀疏初始化比例（默认0.2）
                - diagonal_epsilon: 对角线扰动（默认1e-5，保证数值稳定性）
                - activation: 激活函数（默认'ReLu'）
        """
        super(DeepLagrangianNetwork, self).__init__()

        # Read optional arguments:
        # 读取超参数，如果未提供则使用默认值
        self.n_width = kwargs.get("n_width", 128)  # 隐藏层宽度（神经元数量）
        self.n_hidden = kwargs.get("n_depth", 1)  # 隐藏层数量
        self._b0 = kwargs.get("b_init", 0.1)  # 偏置初始值
        self._b0_diag = kwargs.get("b_diag_init", 0.1)  # 对角元素偏置初始值

        self._w_init = kwargs.get("w_init", "xavier_normal")  # 权重初始化方法
        self._g_hidden = kwargs.get("g_hidden", np.sqrt(2.))  # 隐藏层增益（用于Xavier初始化）
        self._g_output = kwargs.get("g_hidden", 0.125)  # 输出层增益
        self._p_sparse = kwargs.get("p_sparse", 0.2)  # 稀疏初始化的非零比例
        self._epsilon = kwargs.get("diagonal_epsilon", 1.e-5)  # 对角线微小扰动（保证正定性）

        # Construct Weight Initialization:
        # 构造权重初始化函数
        if self._w_init == "xavier_normal":
            # Xavier正态初始化：权重来自N(0, 2/(n_in + n_out))
            # 适合sigmoid/tanh激活函数

            # Construct initialization function:
            # 定义隐藏层初始化函数
            def init_hidden(layer):
                """
                初始化隐藏层权重和偏置
                
                参数:
                    layer: 要初始化的层（包含weight和bias）
                """
                # Set of Hidden Gain:
                # 设置隐藏层增益
                # 如果g_hidden <= 0，使用ReLU的标准增益√2
                # 否则使用用户指定的增益
                if self._g_hidden <= 0.0: 
                    hidden_gain = torch.nn.init.calculate_gain('relu')
                else: 
                    hidden_gain = self._g_hidden

                # 初始化偏置bias为固定值b0
                torch.nn.init.constant_(layer.bias, self._b0)
                # 使用Xavier正态分布初始化权重
                torch.nn.init.xavier_normal_(layer.weight, hidden_gain)

            # 定义输出层初始化函数
            def init_output(layer):
                """
                初始化输出层权重和偏置
                
                参数:
                    layer: 要初始化的层
                """
                # Set Output Gain:
                # 设置输出层增益
                if self._g_output <= 0.0: 
                    output_gain = torch.nn.init.calculate_gain('linear')
                else: 
                    output_gain = self._g_output

                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.xavier_normal_(layer.weight, output_gain)

        elif self._w_init == "orthogonal":
            # 正交初始化：权重矩阵是正交矩阵（W^T×W = I）
            # 有助于梯度流动

            # Construct initialization function:
            def init_hidden(layer):
                if self._g_hidden <= 0.0: 
                    hidden_gain = torch.nn.init.calculate_gain('relu')
                else: 
                    hidden_gain = self._g_hidden

                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.orthogonal_(layer.weight, hidden_gain)

            def init_output(layer):
                if self._g_output <= 0.0: 
                    output_gain = torch.nn.init.calculate_gain('linear')
                else: 
                    output_gain = self._g_output

                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.orthogonal_(layer.weight, output_gain)

        elif self._w_init == "sparse":
            # 稀疏初始化：大部分权重为0，只有p_sparse比例为非零
            # 可以减少计算量和参数数量

            # 断言：检查稀疏比例是否合理
            assert self._p_sparse < 1. and self._p_sparse >= 0.0

            # Construct initialization function:
            def init_hidden(layer):
                p_non_zero = self._p_sparse  # 非零元素比例
                hidden_std = self._g_hidden  # 标准差

                torch.nn.init.constant_(layer.bias, self._b0)
                # 稀疏初始化：只有p_sparse比例的权重非零
                torch.nn.init.sparse_(layer.weight, p_non_zero, hidden_std)

            def init_output(layer):
                p_non_zero = self._p_sparse
                output_std = self._g_output

                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.sparse_(layer.weight, p_non_zero, output_std)

        else:
            # 如果初始化方法不支持，抛出错误
            raise ValueError("Weight Initialization Type must be in ['xavier_normal', 'orthogonal', 'sparse'] but is {0}".format(self._w_init))

        # Compute In- / Output Sizes:
        # 计算输入和输出维度
        self.n_dof = n_dof  # 自由度数
        
        # 下三角矩阵L的非零元素数量
        # 对于n×n的对称矩阵，有n×(n+1)/2个独立元素
        # 下三角矩阵（含对角线）也有这么多元素
        self.m = int((n_dof ** 2 + n_dof) / 2)

        # Compute non-zero elements of L:
        # 计算L矩阵各部分的大小
        l_output_size = int((self.n_dof ** 2 + self.n_dof) / 2)  # 总元素数 = m
        l_lower_size = l_output_size - self.n_dof  # 非对角元素数 = m - n

        # Calculate the indices of the diagonal elements of L:
        # 计算对角元素的索引
        # 例如n=3时，下三角元素的索引顺序是：
        # (0,0)=0, (1,0)=1, (1,1)=2, (2,0)=3, (2,1)=4, (2,2)=5
        # 对角元素索引：0, 2, 5
        # 公式：k = i×(i+1)/2 - 1，其中i从0到n-1
        idx_diag = np.arange(self.n_dof) + 1
        idx_diag = idx_diag * (idx_diag + 1) / 2 - 1

        # Calculate the indices of the off-diagonal elements of L:
        # 计算非对角元素的索引
        # 从所有索引中排除对角元素索引
        idx_tril = np.extract([x not in idx_diag for x in np.arange(l_output_size)], np.arange(l_output_size))

        # Indexing for concatenation of l_o and l_d:
        # 为拼接l_o（非对角）和l_d（对角）创建索引
        # 先放对角元素，再放非对角元素
        cat_idx = np.hstack((idx_diag, idx_tril))
        # 按原始顺序排序，得到新的索引映射
        # 返回从小到大的值的索引
        order = np.argsort(cat_idx)
        # 按列索引（去掉上三角元素）
        self._idx = np.arange(cat_idx.size)[order]

        # create it once and only apply repeat, this may decrease memory allocation
        # 创建单位矩阵（对角线为1，其余为0）
        # shape=(1, n_dof, n_dof)
        self._eye = torch.eye(self.n_dof).view(1, self.n_dof, self.n_dof)
        
        # 创建下三角矩阵转换器
        # 这里是按照行索引的
        self.low_tri = LowTri(self.n_dof)

        # Create Network:
        # 创建神经网络
        self.layers = nn.ModuleList()  # 存储所有层的列表
        non_linearity = kwargs.get("activation", "ReLu")  # 激活函数类型

        # Create Input Layer:
        # 创建输入层：从n_dof维到n_width维
        self.layers.append(LagrangianLayer(self.n_dof, self.n_width, activation=non_linearity))
        init_hidden(self.layers[-1])  # 初始化该层

        # Create Hidden Layer:
        # 创建隐藏层（可能有多个）
        for _ in range(1, self.n_hidden):
            self.layers.append(LagrangianLayer(self.n_width, self.n_width, activation=non_linearity))
            init_hidden(self.layers[-1])

        # Create output Layer:
        # 创建输出层（三个头）

        # 1. 势能网络：输出V(q)（标量）
        # 使用线性激活（不限制输出范围）
        self.net_g = LagrangianLayer(self.n_width, 1, activation="Linear")
        init_output(self.net_g)

        # 2. L矩阵非对角元素网络：输出l_lower（向量）
        # 使用线性激活（非对角元素可以为负）
        self.net_lo = LagrangianLayer(self.n_width, l_lower_size, activation="Linear")
        init_hidden(self.net_lo)

        # 3. L矩阵对角元素网络：输出l_diag（向量）
        # The diagonal must be non-negative. Therefore, non-linearity is set to ReLu.
        # 对角元素必须为正（保证H正定），所以用ReLU激活
        self.net_ld = LagrangianLayer(self.n_width, self.n_dof, activation="ReLu")
        init_hidden(self.net_ld)
        # 对角元素的偏置初始化为较大的正值
        torch.nn.init.constant_(self.net_ld.bias, self._b0_diag)

    def forward(self, q, qd, qdd):
        """
        主前向传播函数
        
        参数:
            q (torch.Tensor): 关节位置，shape=(batch_size, n_dof)
            qd (torch.Tensor): 关节速度，shape=(batch_size, n_dof)
            qdd (torch.Tensor): 关节加速度，shape=(batch_size, n_dof)
        
        返回:
            tau_pred (torch.Tensor): 预测的关节力矩，shape=(batch_size, n_dof)
            dEdt (torch.Tensor): 能量变化率（功率），shape=(batch_size,)
        
        物理意义：
        - 通过拉格朗日方程计算力矩：τ = H×q̈ + c + g
        - 功率守恒：dE/dt = τ^T × q̇
        """
        # 调用内部动力学模型，计算所有物理量
        out = self._dyn_model(q, qd, qdd)
        
        # 提取预测的力矩（第0个输出）
        tau_pred = out[0]
        
        # 计算总能量变化率 = 动能变化率 + 势能变化率
        dEdt = out[6] + out[7]

        return tau_pred, dEdt

    def _dyn_model(self, q, qd, qdd):
        """
        内部动力学模型（核心计算函数）
        
        计算拉格朗日方程中的所有物理量：
        1. 惯性矩阵H及其导数
        2. 科里奥利力c
        3. 重力g
        4. 动能T和势能V
        5. 能量变化率dT/dt和dV/dt
        
        参数:
            q (torch.Tensor): 位置
            qd (torch.Tensor): 速度
            qdd (torch.Tensor): 加速度（仅用于力矩计算，能量计算时设为0）
        
        返回:
            (tau_pred, H, c, g, T, V, dTdt, dVdt): 所有物理量的元组
        """
        # 将速度张量重塑为3D和4D，方便后续计算
        # qd_3d: (batch, n_dof, 1) - 用于3D矩阵乘法
        # qd_4d: (batch, 1, n_dof, 1) - 用于4D张量运算
        qd_3d = qd.view(-1, self.n_dof, 1)
        qd_4d = qd.view(-1, 1, self.n_dof, 1)

        # Create initial derivative of dq/dq.
        # 创建初始的雅可比矩阵：∂q/∂q = I（单位矩阵）
        # 这是链式法则的起始点：∂y₀/∂q = I
        der = self._eye.repeat(q.shape[0], 1, 1).type_as(q)

        # Compute shared network between l & g:
        # 计算共享网络的输出（主干网络）
        # 这部分网络同时服务于L矩阵和势能V
        
        # 通过第一层（输入层）
        y, der = self.layers[0](q, der)
        # y: 隐藏层输出
        # der: 输出对输入的雅可比矩阵

        # 通过后续隐藏层
        for i in range(1, len(self.layers)):
            y, der = self.layers[i](y, der)

        # Compute the network heads including the corresponding derivative:
        # 计算三个输出头（包括它们的导数）
        
        # 1. L矩阵非对角元素网络
        l_lower, der_l_lower = self.net_lo(y, der)
        # l_lower: 非对角元素向量
        # der_l_lower: ∂l_lower/∂q（雅可比矩阵）

        # 2. L矩阵对角元素网络
        l_diag, der_l_diag = self.net_ld(y, der)
        # l_diag: 对角元素向量
        # der_l_diag: ∂l_diag/∂q（雅可比矩阵）

        # 3. 势能网络
        # Compute potential energy and its derivative:
        # 计算势能V(q)和重力g(q) = ∂V/∂q
        V, der_V = self.net_g(y, der)
        # V: 势能（标量）
        # der_V: 势能的梯度 = 重力（向量）
        V = V.squeeze()  # 去除多余的维度
        g = der_V.squeeze()  # 重力g = ∇V

        # Assemble l and der_l
        # 组合L矩阵的元素（对角+非对角）
        # 先放对角，再放非对角，然后按原始顺序重排
        l_diag = l_diag
        l = torch.cat((l_diag, l_lower), 1)[:, self._idx]  # 拼接并重排序
        der_l = torch.cat((der_l_diag, der_l_lower), 1)[:, self._idx, :]  # 导数也拼接并重排

        # Compute H:
        # 计算惯性矩阵H = L × L^T（Cholesky分解）
        # 这样保证H正定对称
        L = self.low_tri(l)  # 将向量转为下三角矩阵
        LT = L.transpose(dim0=1, dim1=2)  # 转置：L^T
        # H = L × L^T + ε×I
        # ε×I保证H严格正定（数值稳定性）
        H = torch.matmul(L, LT) + self._epsilon * torch.eye(self.n_dof).type_as(L)

        # Calculate dH/dt
        # 计算H对时间的导数
        # H = L(q) × L(q)^T，所以 dH/dt = dL/dt × L^T + L × dL/dt^T
        # dL/dt = (∂L/∂q) × dq/dt = der_l × qd
        
        # 计算 dL/dt 的向量形式
        # der_l: ∂l/∂q，shape=(batch, m, n_dof)
        # qd_3d: q̇，shape=(batch, n_dof, 1)
        # matmul: (batch, m, n_dof) × (batch, n_dof, 1) = (batch, m, 1)
        Ldt = self.low_tri(torch.matmul(der_l, qd_3d).view(-1, self.m))
        
        # dH/dt = L × dL/dt^T + dL/dt × L^T
        Hdt = torch.matmul(L, Ldt.transpose(dim0=1, dim1=2)) + torch.matmul(Ldt, LT)

        # Calculate dH/dq:
        # 计算H对位置的偏导数（4D张量）
        # dH/dq 是一个3D张量：对每个q分量，有一个H矩阵
        # shape: (batch, n_dof, n_dof, n_dof)
        # H[i,j,k] = ∂H_ij/∂q_k
        
        # dL/dq: shape=(batch, m, n_dof, n_dof)
        Ldq = self.low_tri(der_l.transpose(2, 1).reshape(-1, self.m)).reshape(-1, self.n_dof, self.n_dof, self.n_dof)
        
        # dH/dq = dL/dq × L^T + L × dL/dq^T
        Hdq = torch.matmul(Ldq, LT.view(-1, 1, self.n_dof, self.n_dof)) + \
               torch.matmul(L.view(-1, 1, self.n_dof, self.n_dof), Ldq.transpose(2, 3))

        # Compute Coriolis & Centrifugal forces:
        # 计算科里奥利力和离心力
        # 公式：c = dH/dt × q̇ - 0.5 × q̇^T × dH/dq × q̇
        # 第一项：dH/dt × q̇
        Hdt_qd = torch.matmul(Hdt, qd_3d).view(-1, self.n_dof)
        
        # 第二项：0.5 × q̇^T × dH/dq × q̇
        # qd_4d^T: (batch, 1, 1, n_dof)
        # dH/dq: (batch, n_dof, n_dof, n_dof)
        # qd_4d: (batch, 1, n_dof, 1)
        quad_dq = torch.matmul(qd_4d.transpose(dim0=2, dim1=3), 
                          torch.matmul(Hdq, qd_4d)).view(-1, self.n_dof)
        
        # 科里奥利力 = 第一项 - 第二项
        c = Hdt_qd - 1. / 2. * quad_dq

        # Compute the Torque using inverse model:
        # 使用逆动力学模型计算力矩
        # 拉格朗日方程：τ = H × q̈ + c + g
        # 1. H × q̈（惯性力）
        H_qdd = torch.matmul(H, qdd.view(-1, self.n_dof, 1)).view(-1, self.n_dof)
        # 2. 总力矩 = 惯性力 + 科里奥利力 + 重力
        tau_pred = H_qdd + c + g

        # Compute kinetic energy T
        # 计算动能 T = 0.5 × q̇^T × H × q̇
        H_qd = torch.matmul(H, qd_3d).view(-1, self.n_dof)
        T = 1. / 2. * torch.matmul(qd_4d.transpose(dim0=2, dim1=3), 
                                H_qd.view(-1, 1, self.n_dof, 1)).view(-1)

        # Compute dT/dt:
        # 计算动能对时间的导数
        # dT/dt = q̇^T × H × q̈ + 0.5 × q̇^T × dH/dt × q̇
        
        # 第一项：q̇^T × H × q̈
        qd_H_qdd = torch.matmul(qd_4d.transpose(dim0=2, dim1=3), 
                             H_qdd.view(-1, 1, self.n_dof, 1)).view(-1)
        
        # 第二项：0.5 × q̇^T × dH/dt × q̇
        qd_Hdt_qd = torch.matmul(qd_4d.transpose(dim0=2, dim1=3), 
                              Hdt_qd.view(-1, 1, self.n_dof, 1)).view(-1)
        
        dTdt = qd_H_qdd + 0.5 * qd_Hdt_qd

        # Compute dV/dt
        # 计算势能对时间的导数
        # dV/dt = ∂V/∂q × dq/dt = g^T × q̇
        dVdt = torch.matmul(qd_4d.transpose(dim0=2, dim1=3), 
                         g.view(-1, 1, self.n_dof, 1)).view(-1)
        
        # 返回所有物理量：
        # 0. tau_pred: 预测力矩
        # 1. H: 惯性矩阵
        # 2. c: 科里奥利力
        # 3. g: 重力
        # 4. T: 动能
        # 5. V: 势能
        # 6. dTdt: 动能变化率
        # 7. dVdt: 势能变化率
        return tau_pred, H, c, g, T, V, dTdt, dVdt

    def inv_dyn(self, q, qd, qdd):
        """
        逆动力学：已知运动，求力矩
        
        参数:
            q: 位置
            qd: 速度
            qdd: 加速度
        
        返回:
            tau_pred: 预测的关节力矩
        """
        out = self._dyn_model(q, qd, qdd)
        tau_pred = out[0]
        return tau_pred

    def for_dyn(self, q, qd, tau):
        """
        正动力学：已知力矩，求加速度
        
        参数:
            q: 位置
            qd: 速度
            tau: 关节力矩
        
        返回:
            qdd_pred: 预测的关节加速度
        """
        # 计算动力学量（加速度设为0，因为此时不需要）
        out = self._dyn_model(q, qd, torch.zeros_like(q))
        H, c, g = out[1], out[2], out[3]

        # Compute Acceleration, e.g., forward model:
        # 计算加速度（正动力学）
        # 从拉格朗日方程：τ = H × q̈ + c + g
        # 解得：q̈ = H^(-1) × (τ - c - g)
        
        # 1. 求H的逆矩阵
        invH = torch.inverse(H)
        
        # 2. 计算有效力矩（减去科里奥利力和重力）
        effective_tau = tau - c - g
        
        # 3. 计算加速度：q̈ = H^(-1) × (τ - c - g)
        qdd_pred = torch.matmul(invH, effective_tau.view(-1, self.n_dof, 1)).view(-1, self.n_dof)
        return qdd_pred

    def energy(self, q, qd):
        """
        计算总能量（动能 + 势能）
        
        参数:
            q: 位置
            qd: 速度
        
        返回:
            E: 总能量
        """
        out = self._dyn_model(q, qd, torch.zeros_like(q))
        # 总能量 = 动能 + 势能 = T + V
        E = out[4] + out[5]
        return E

    def energy_dot(self, q, qd, qdd):
        """
        计算能量变化率（功率）
        
        根据能量守恒：dE/dt = τ^T × q̇
        即功率 = 力矩 · 速度
        
        参数:
            q: 位置
            qd: 速度
            qdd: 加速度
        
        返回:
            dEdt: 能量变化率
        """
        out = self._dyn_model(q, qd, qdd)
        # 能量变化率 = 动能变化率 + 势能变化率
        dEdt = out[6] + out[7]
        return dEdt

    def cuda(self, device=None):
        """
        将模型移动到GPU
        
        参数:
            device: GPU设备ID，如果为None则使用默认设备
        
        返回:
            self: 模型对象本身
        """
        # Move the Network to GPU:
        # 将神经网络移动到GPU
        super(DeepLagrangianNetwork, self).cuda(device=device)

        # Move the eye matrix to GPU:
        # 将单位矩阵也移动到GPU（用于初始化雅可比矩阵）
        self._eye = self._eye.cuda()
        self.device = self._eye.device  # 记录当前设备
        return self

    def cpu(self):
        """
        将模型移动到CPU
        
        返回:
            self: 模型对象本身
        """
        # Move the Network to CPU:
        # 将神经网络移动到CPU
        super(DeepLagrangianNetwork, self).cpu()

        # Move the eye matrix to CPU:
        # 将单位矩阵也移动到CPU
        self._eye = self._eye.cpu()
        self.device = self._eye.device  # 记录当前设备
        return self
