"""
深度拉格朗日网络工具函数库
包含环境初始化、数据加载、参数处理等辅助函数
"""

import sys

import dill as pickle  # 使用dill库代替标准pickle，支持更多Python对象
import numpy as np  # NumPy：数值计算库，用于数组操作
import torch  # PyTorch：深度学习框架
import jax  # JAX：函数式机器学习框架
import jax.numpy as jnp  # JAX的NumPy接口
import haiku as hk  # Haiku：JAX的神经网络库
import functools  # functools：高阶函数和工具函数


def init_env(args):
    """
    初始化实验环境
    
    参数:
        args: 命令行参数对象，包含:
            args.s[0]: 随机种子
            args.i[0]: CUDA设备ID
            args.c[0]: 是否使用CUDA的标志
            args.r[0]: 是否渲染/可视化的标志
            args.l[0]: 是否加载预训练模型的标志
            args.m[0]: 是否保存模型的标志
    
    返回:
        seed: 随机种子
        cuda_flag: 是否使用CUDA（如果可用）
        render: 是否渲染/可视化
        load_model: 是否加载预训练模型
        save_model: 是否保存模型
    """

    # 设置NumPy的打印格式：不显示科学计数法，保留2位小数，行宽500字符
    # 这让数值输出更易读，例如：0.001234 → 0.00
    np.set_printoptions(suppress=True, precision=2, linewidth=500,
                        formatter={'float_kind': lambda x: "{0:+08.2f}".format(x)})

    # 从参数对象中提取各个配置项
    # args.s[0]: 随机种子，用于确保实验可重复性
    # args.i[0]: GPU设备ID，当有多张GPU时选择使用哪一张
    # args.c[0]: 是否使用CUDA的布尔标志
    seed, cuda_id, cuda_flag = args.s[0], args.i[0], args.c[0]
    
    # 将数字转换为布尔值（Python中bool(1)=True, bool(0)=False）
    render, load_model, save_model = bool(args.r[0]), bool(args.l[0]), bool(args.m[0])

    # 只有当cuda_flag为True且系统确实有CUDA可用时，才使用CUDA
    # torch.cuda.is_available() 检查系统中是否有可用的NVIDIA GPU
    cuda_flag = cuda_flag and torch.cuda.is_available()

    # 设置随机种子，确保实验结果可重复
    # 如果种子相同，每次运行生成的随机数序列就相同
    np.random.seed(seed)  # 设置NumPy的随机种子
    torch.manual_seed(seed)  # 设置PyTorch CPU的随机种子
    torch.cuda.manual_seed_all(seed)  # 设置所有GPU的随机种子

    # 配置CUDA设备
    if torch.cuda.device_count() > 1:  # 如果有多张GPU
        # 确保选择的设备ID有效（不能超过GPU总数）
        assert cuda_id < torch.cuda.device_count()
        # 设置当前使用的GPU设备
        torch.cuda.set_device(cuda_id)

    return seed, cuda_flag, render, load_model, save_model


def load_dataset(n_characters=3, filename="data/character_data.pickle", test_label=("e", "q", "v")):
    """
    加载并分割训练/测试数据集
    
    参数:
        n_characters: 测试集中包含的字符数量（未使用，保留参数）
        filename: 数据文件路径，pickle格式
        test_label: 用作测试集的字符标签元组，例如 ("e", "q", "v")。
            若某标签不在数据集中会忽略并打印警告；若最终一个都匹配不到，则
            将全部轨迹同时放入训练集与测试集（脚本可运行，但不再有留出泛化）。
    
    返回:
        train_data: 训练数据元组
            (train_labels, train_qp, train_qv, train_qa, train_p, train_pd, train_tau)
        test_data: 测试数据元组
            (test_labels, test_qp, test_qv, test_qa, test_p, test_pd, test_tau, test_m, test_c, test_g)
        divider: 分隔数组，用于绘图时区分不同字符的数据点
        dt_mean: 平均时间步长
    
    物理量说明:
        qp (q position): 关节位置坐标
        qv (q velocity): 关节速度
        qa (q acceleration): 关节加速度
        tau: 关节力矩（控制输入）
        m (mass/inertial): 惯性力
        c (coriolis): 科里奥利力和离心力
        g (gravity): 重力
        p (momentum): 动量
        pdot (momentum dot): 动量的时间导数

    自由度 n_dof 由每条轨迹 qp 的列数推断（所有轨迹必须相同）。
    """

    # 从pickle文件中加载数据
    # 'rb' 模式表示以二进制读取
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    # 从数据推断关节维；各轨迹必须一致
    n_dof = int(np.asarray(data["qp"][0]).shape[1])
    for i in range(len(data["labels"])):
        d_i = int(np.asarray(data["qp"][i]).shape[1])
        if d_i != n_dof:
            raise ValueError(
                f"数据集中各轨迹 n_dof 不一致: 轨迹0为{n_dof}，轨迹{i}为{d_i}"
            )

    # 将数据集分割为训练集和测试集

    # 方法1: 随机选择测试集（未使用）
    # test_idx = np.random.choice(len(data["labels"]), n_characters, replace=False)

    # 方法2: 指定特定字符作为测试集；跳过 pickle 里不存在的标签
    labels_all = data["labels"]
    test_idx = []
    missing = []
    for x in test_label:
        if x in labels_all:
            ix = labels_all.index(x)
            if ix not in test_idx:
                test_idx.append(ix)
        else:
            missing.append(x)
    if missing:
        print(
            "load_dataset 警告: 以下标签不在数据中，已从测试划分中忽略: "
            f"{missing}",
            file=sys.stderr,
        )
    use_full_overlap = False
    if not test_idx:
        print(
            "load_dataset 警告: 无任何请求的测试标签存在于数据中；"
            "将全部轨迹同时划入训练集与测试集（训练仍可运行，但不是留出字符泛化）。",
            file=sys.stderr,
        )
        test_idx = list(range(len(labels_all)))
        use_full_overlap = True

    # 计算所有测试轨迹的时间步长（相邻时间点的差值）
    # data["t"]: 时间数组
    # [1:] - [:-1]: 取后n-1个元素减去前n-1个元素，得到时间间隔
    # np.concatenate: 将多个轨迹的时间步长连接成一个数组
    dt = np.concatenate([data["t"][idx][1:] - data["t"][idx][:-1] for idx in test_idx])
    
    # 计算时间步长的均值和方差
    dt_mean, dt_var = np.mean(dt), np.var(dt)
    
    # 断言：检查时间步长是否恒定（方差应该接近0）
    # 这要求使用固定时间步长的仿真数据
    assert dt_var < 1.e-12

    # 初始化训练数据的数组（0行n_dof列）
    train_labels, test_labels = [], []  # 标签列表
    # np.zeros((行, 列))
    train_qp, train_qv, train_qa, train_tau = np.zeros((0, n_dof)), np.zeros((0, n_dof)), np.zeros((0, n_dof)), np.zeros((0, n_dof))
    train_p, train_pd = np.zeros((0, n_dof)), np.zeros((0, n_dof))

    # 初始化测试数据的数组
    test_qp, test_qv, test_qa, test_tau = np.zeros((0, n_dof)), np.zeros((0, n_dof)), np.zeros((0, n_dof)), np.zeros((0, n_dof))
    # 这里分别表示测试集的真实物理量：惯性力、科里奥利力、重力
    test_m, test_c, test_g = np.zeros((0, n_dof)), np.zeros((0, n_dof)), np.zeros((0, n_dof))  # 测试集还有真实的物理量用于对比
    test_p, test_pd = np.zeros((0, n_dof)), np.zeros((0, n_dof))

    # 分隔数组：用于在绘图时区分不同字符的数据点
    # 记录每个字符在测试集中数据点的累计数量
    divider = [0, ]

    # 遍历所有字符轨迹
    for i in range(len(data["labels"])):

        in_test = i in test_idx
        in_train = use_full_overlap or (i not in test_idx)

        if in_test:  # 如果当前字符在测试集中
            # 将测试数据添加到测试集数组
            test_labels.append(data["labels"][i])
            
            # np.vstack: 垂直堆叠数组，将新数据追加到现有数组下方
            # 例如：数组A shape=(10,2)，数组B shape=(5,2)，vstack后shape=(15,2)
            test_qp = np.vstack((test_qp, data["qp"][i]))  # 添加位置数据
            test_qv = np.vstack((test_qv, data["qv"][i]))  # 添加速度数据
            test_qa = np.vstack((test_qa, data["qa"][i]))  # 添加加速度数据
            test_tau = np.vstack((test_tau, data["tau"][i]))  # 添加力矩数据

            test_m = np.vstack((test_m, data["m"][i]))  # 添加真实的惯性力
            test_c = np.vstack((test_c, data["c"][i]))  # 添加真实的科里奥利力
            test_g = np.vstack((test_g, data["g"][i]))  # 添加真实的重力

            test_p = np.vstack((test_p, data["p"][i]))  # 添加动量数据
            test_pd = np.vstack((test_pd, data["pdot"][i]))  # 添加动量导数数据
            
            # 记录当前字符数据点的累计数量，用于后续绘图时的分隔
            # test_qp.shape[0]：总行数（数据长度）
            divider.append(test_qp.shape[0])

        if in_train:  # 训练集：留出划分；或与测试集完全重叠（use_full_overlap）
            # 将训练数据添加到训练集数组
            train_labels.append(data["labels"][i])
            train_qp = np.vstack((train_qp, data["qp"][i]))
            train_qv = np.vstack((train_qv, data["qv"][i]))
            train_qa = np.vstack((train_qa, data["qa"][i]))
            train_tau = np.vstack((train_tau, data["tau"][i]))

            train_p = np.vstack((train_p, data["p"][i]))
            train_pd = np.vstack((train_pd, data["pdot"][i]))

    if len(train_labels) == 0 and test_qp.shape[0] > 0:
        print(
            "load_dataset 警告: 训练集为空（例如仅指定了测试标签且数据只有这些轨迹）。"
            "已把测试集内容复制到训练集以便能训练。",
            file=sys.stderr,
        )
        train_labels = list(test_labels)
        train_qp = np.array(test_qp, copy=True)
        train_qv = np.array(test_qv, copy=True)
        train_qa = np.array(test_qa, copy=True)
        train_tau = np.array(test_tau, copy=True)
        train_p = np.array(test_p, copy=True)
        train_pd = np.array(test_pd, copy=True)

    # 返回三个值：训练数据、测试数据、分隔数组、平均时间步长
    return (train_labels, train_qp, train_qv, train_qa, train_p, train_pd, train_tau), \
           (test_labels, test_qp, test_qv, test_qa, test_p, test_pd, test_tau, test_m, test_c, test_g),\
           divider, dt_mean


def parition_params(module_name, name, value, key):
    """
    参数分区辅助函数：判断参数是否属于特定模块
    
    参数:
        module_name: 完整的模块路径（如 "network/layers/hidden"）
        name: 参数名称
        value: 参数值
        key: 要匹配的模块关键字
    
    返回:
        bool: 如果模块路径以key开头，返回True；否则返回False
    
    用途:
        用于将神经网络参数按模块分组
        例如：将参数分为"编码器"和"解码器"两组
    """
    # module_name.split("/")[0]: 获取模块路径的第一部分
    # 例如："network/layers/hidden" → "network"
    return module_name.split("/")[0] == key


def get_params(params, key):
    """
    从参数字典中提取特定模块的参数
    
    参数:
        params: 完整的参数字典
        key: 要提取的模块关键字
    
    返回:
        特定模块的参数字典
    
    用途:
        用于获取神经网络中某一部分的参数
        例如：只获取编码器的参数，用于单独训练或分析
    """
    # 使用Haiku库的partition函数，将参数按条件分区
    # functools.partial: 创建偏函数，固定parition_params的key参数
    return hk.data_structures.partition(functools.partial(parition_params, key=key), params)


# 激活函数字典：将字符串映射到实际的激活函数
activations = {
    'tanh': jnp.tanh,         # 双曲正切函数，输出范围[-1, 1]
    'softplus': jax.nn.softplus,  # Softplus函数，平滑的ReLU变体，输出范围(0, ∞)
}
