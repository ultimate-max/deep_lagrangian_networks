#!/usr/bin/env python3
"""
将 MATLAB 导出的轨迹数据（.mat）转换为 DeLaN 使用的 character_data.pickle。

建议在 DeLaN conda 环境中运行：
    conda activate DeLaN
    python tools/mat_to_character_pickle.py -i your_data.mat -o data/character_data.pickle

形式 A — 顶层为各变量：labels, t, qp, ...（见下方）

形式 B — 全部包在一个 struct 里（如 character_data），可用 --root；若 .mat
    仅含一个顶层 struct 则自动展开。

形式 C — 单条轨迹、无 cell：labels 为单个字符；t 为 T×1；qp 等为 T×n_dof 数值矩阵

变量约定（推荐 save -v7 以便 scipy 读取）：
    labels        1xN cell 或 单个字符
    t, qp, qv, qa, tau, m, c, g, p, pdot
    mass_matrix   （可选）

示例（MATLAB，形式 A）：
    labels = {'a','b'};
    t = {t1, t2};
    qp = {qp1, qp2};
    % ... 填齐 qv, qa, tau, m, c, g, p, pdot ...
    save('exported.mat', 'labels','t','qp','qv','qa','tau','m','c','g','p','pdot','-v7');
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import dill as pickle
import numpy as np
from scipy.io import loadmat


def _is_mat_struct(obj) -> bool:
    return type(obj).__name__ == "mat_struct"


def _mat_struct_to_dict(obj) -> dict:
    out = {}
    for name in dir(obj):
        if name.startswith("_"):
            continue
        val = getattr(obj, name)
        if callable(val):
            continue
        out[name] = val
    return out


def _resolve_mat_payload(mat: dict, root: str | None) -> dict:
    mat = _strip_mat_meta(mat)
    if root is not None:
        if root not in mat:
            raise KeyError(f".mat 中找不到结构体变量: {root!r}")
        obj = mat[root]
        if not _is_mat_struct(obj):
            raise TypeError(f"{root!r} 不是 MATLAB struct（mat_struct）")
        return _mat_struct_to_dict(obj)
    if len(mat) == 1:
        sole = next(iter(mat.values()))
        if _is_mat_struct(sole):
            return _mat_struct_to_dict(sole)
    return mat


def _strip_mat_meta(mat: dict) -> dict:
    out = {}
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        out[k] = v
    return out


def _infer_n_dof_from_qp(qp: np.ndarray, transpose: bool) -> int:
    a = np.asarray(qp, dtype=np.float64)
    if transpose:
        a = a.T
    if a.ndim != 2:
        raise ValueError(f"推断 n_dof 需要 qp 为二维矩阵，得到 shape={a.shape}")
    r, c = a.shape
    if r == c:
        raise ValueError(
            f"无法自动推断 n_dof：qp 为方阵 {a.shape}，请显式传入 --n-dof"
        )
    return int(r if r < c else c)


def _as_trajectory_list(arr: np.ndarray) -> list[np.ndarray]:
    if arr.dtype != object:
        raise ValueError(
            "期望 MATLAB cell 数组（numpy dtype=object）；若为数值矩阵请用无 cell 单轨迹格式。"
        )
    flat = arr.ravel(order="F") if arr.ndim > 1 else arr
    return [np.asarray(flat[i]).squeeze() for i in range(flat.size)]


def _as_label_list(raw) -> list[str]:
    if isinstance(raw, np.ndarray):
        if raw.dtype == object:
            items = raw.ravel(order="F") if raw.ndim > 1 else raw
            return [_normalize_label(items[i]) for i in range(items.size)]
        if raw.dtype.kind in ("U", "S", "O"):
            flat = raw.ravel(order="F")
            return [_normalize_label(flat[i]) for i in range(flat.size)]
    return [_normalize_label(raw)]


def _normalize_label(x) -> str:
    if isinstance(x, bytes):
        return x.decode("ascii")
    if isinstance(x, str):
        return x if len(x) <= 1 else x.strip()[:1]
    s = str(np.asarray(x).item())
    return s[:1] if s else "?"


def _maybe_transpose_joints(a: np.ndarray, n_dof: int, transpose: bool) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"期望轨迹矩阵为 2 维，得到 shape={a.shape}")
    if transpose:
        a = a.T
    if a.shape[0] == n_dof and a.shape[1] != n_dof:
        a = a.T
    if a.ndim != 2 or a.shape[1] != n_dof:
        raise ValueError(
            f"形状 {a.shape} 与 n_dof={n_dof} 不符；期望 (T, {n_dof})，可使用 --transpose"
        )
    return np.ascontiguousarray(a)


def _maybe_transpose_mass(a: np.ndarray, n_dof: int) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 3:
        raise ValueError(f"mass_matrix 期望 3 维，得到 shape={a.shape}")
    if a.shape[0] == n_dof and a.shape[1] == n_dof and a.shape[2] != n_dof:
        a = np.transpose(a, (2, 0, 1))
    if a.shape[1] != n_dof or a.shape[2] != n_dof:
        raise ValueError(
            f"mass_matrix 期望末尾两维为 ({n_dof},{n_dof})，得到 {a.shape}"
        )
    return np.ascontiguousarray(a)


def _coerce_to_per_trajectory_lists(mat: dict) -> tuple[list[str], dict[str, list], list | None]:
    required = ("labels", "t", "qp", "qv", "qa", "tau", "m", "c", "g", "p", "pdot")
    for k in required:
        if k not in mat:
            raise KeyError(f".mat 缺少变量: {k}")

    qp = mat["qp"]
    is_cell = isinstance(qp, np.ndarray) and qp.dtype == object

    if is_cell:
        labels = _as_label_list(mat["labels"])
        n_traj = len(labels)

        def series(key: str) -> list[np.ndarray]:
            return _as_trajectory_list(np.asarray(mat[key], dtype=object))

        t_list = series("t")
        qp_list = series("qp")
        qv_list = series("qv")
        qa_list = series("qa")
        tau_list = series("tau")
        m_list = series("m")
        c_list = series("c")
        g_list = series("g")
        p_list = series("p")
        pdot_list = series("pdot")

        lengths = [
            n_traj,
            len(t_list),
            len(qp_list),
            len(qv_list),
            len(qa_list),
            len(tau_list),
            len(m_list),
            len(c_list),
            len(g_list),
            len(p_list),
            len(pdot_list),
        ]
        if len(set(lengths)) != 1:
            raise ValueError(f"各字段轨迹条数不一致: {lengths}")

        mm_list = None
        if "mass_matrix" in mat and mat["mass_matrix"] is not None:
            mm_list = _as_trajectory_list(np.asarray(mat["mass_matrix"], dtype=object))
            if len(mm_list) != n_traj:
                raise ValueError("mass_matrix 轨迹条数与其它字段不一致")

        series_dict = {
            "t": t_list,
            "qp": qp_list,
            "qv": qv_list,
            "qa": qa_list,
            "tau": tau_list,
            "m": m_list,
            "c": c_list,
            "g": g_list,
            "p": p_list,
            "pdot": pdot_list,
        }
        return labels, series_dict, mm_list

    if not isinstance(qp, np.ndarray) or qp.ndim != 2:
        raise ValueError(
            "qp 既不是 MATLAB cell（dtype=object），也不是二维数值矩阵；无法解析。"
        )
    labels = _as_label_list(mat["labels"])
    if len(labels) != 1:
        raise ValueError(
            "当前为「无 cell 的矩阵格式」，仅支持单条轨迹；labels 应为单个字符，"
            "或将每条轨迹用 cell 分开保存。得到 labels="
            f"{labels}"
        )
    t_list = [np.asarray(mat["t"], dtype=np.float64).reshape(-1)]
    keys = ("qp", "qv", "qa", "tau", "m", "c", "g", "p", "pdot")
    series_dict = {
        "t": t_list,
        **{k: [np.asarray(mat[k], dtype=np.float64)] for k in keys},
    }
    mm_list = None
    if "mass_matrix" in mat and mat["mass_matrix"] is not None:
        mm_list = [np.asarray(mat["mass_matrix"], dtype=np.float64)]
    return labels, series_dict, mm_list


def mat_to_pickle_dict(
    mat: dict, n_dof: int | None, transpose: bool, root: str | None
) -> dict:
    mat = _resolve_mat_payload(mat, root)
    labels, series_dict, mm_list = _coerce_to_per_trajectory_lists(mat)
    n_traj = len(labels)

    if n_dof is None:
        n_dof = _infer_n_dof_from_qp(series_dict["qp"][0], transpose)

    t_list = series_dict["t"]
    qp_list = series_dict["qp"]
    qv_list = series_dict["qv"]
    qa_list = series_dict["qa"]
    tau_list = series_dict["tau"]
    m_list = series_dict["m"]
    c_list = series_dict["c"]
    g_list = series_dict["g"]
    p_list = series_dict["p"]
    pdot_list = series_dict["pdot"]

    out: dict = {"labels": labels}
    out["t"] = []
    out["qp"] = []
    out["qv"] = []
    out["qa"] = []
    out["tau"] = []
    out["m"] = []
    out["c"] = []
    out["g"] = []
    out["p"] = []
    out["pdot"] = []
    if mm_list is not None:
        out["mass_matrix"] = []

    for i in range(n_traj):
        qp_i = _maybe_transpose_joints(qp_list[i], n_dof, transpose)
        qv_i = _maybe_transpose_joints(qv_list[i], n_dof, transpose)
        qa_i = _maybe_transpose_joints(qa_list[i], n_dof, transpose)
        tau_i = _maybe_transpose_joints(tau_list[i], n_dof, transpose)
        m_i = _maybe_transpose_joints(m_list[i], n_dof, transpose)
        c_i = _maybe_transpose_joints(c_list[i], n_dof, transpose)
        g_i = _maybe_transpose_joints(g_list[i], n_dof, transpose)
        p_i = _maybe_transpose_joints(p_list[i], n_dof, transpose)
        pd_i = _maybe_transpose_joints(pdot_list[i], n_dof, transpose)

        Ti = qp_i.shape[0]
        t_i = np.asarray(t_list[i], dtype=np.float64).reshape(-1)
        if t_i.shape[0] != Ti:
            raise ValueError(
                f"轨迹 {i}: 时间向量长度 {t_i.shape[0]} 与 qp 行数 {Ti} 不一致"
            )
        rows = [
            qp_i.shape[0],
            qv_i.shape[0],
            qa_i.shape[0],
            tau_i.shape[0],
            m_i.shape[0],
            c_i.shape[0],
            g_i.shape[0],
            p_i.shape[0],
            pd_i.shape[0],
        ]
        if len(set(rows)) != 1:
            raise ValueError(f"轨迹 {i}: 各序列时间步 T 不一致: {rows}")

        out["t"].append(t_i)
        out["qp"].append(qp_i)
        out["qv"].append(qv_i)
        out["qa"].append(qa_i)
        out["tau"].append(tau_i)
        out["m"].append(m_i)
        out["c"].append(c_i)
        out["g"].append(g_i)
        out["p"].append(p_i)
        out["pdot"].append(pd_i)

        if mm_list is not None:
            mm_i = _maybe_transpose_mass(mm_list[i], n_dof)
            if mm_i.shape[0] != Ti:
                raise ValueError(f"轨迹 {i}: mass_matrix 第 0 维应等于 T={Ti}")
            out["mass_matrix"].append(mm_i)

    means = []
    max_var = 0.0
    for t_i in out["t"]:
        if t_i.size < 2:
            continue
        d = np.diff(t_i)
        means.append(float(np.mean(d)))
        max_var = max(max_var, float(np.var(d)))
    if len(means) > 1:
        if max(abs(means[i] - means[0]) for i in range(1, len(means))) > 1e-9:
            raise ValueError(
                "不同轨迹的平均 dt 不一致；load_dataset 假定全局恒定步长。"
            )
    if max_var >= 1e-12:
        print(
            "警告: 至少一条轨迹内时间间隔方差较大，utils.load_dataset 可能 assert 失败。",
            file=sys.stderr,
        )

    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MATLAB .mat → character_data.pickle（DeLaN load_dataset 格式）"
    )
    parser.add_argument(
        "-i", "--input", type=Path, required=True, help="MATLAB 保存的 .mat 路径"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="输出 pickle 路径，例如 data/character_data.pickle",
    )
    parser.add_argument(
        "--n-dof",
        type=int,
        default=None,
        help="关节自由度；省略时按第一条轨迹 qp 的形状自动推断",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="MATLAB 中存放字段的 struct 名（如 character_data）；若 .mat 仅含一个顶层 struct 则可省略",
    )
    parser.add_argument(
        "--transpose",
        action="store_true",
        help="对每条轨迹的 qp/qv/... 先转置再解析（MATLAB 存成 n_dof×T 时常用）",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"找不到输入文件: {args.input}", file=sys.stderr)
        return 1

    mat = loadmat(
        args.input, squeeze_me=True, struct_as_record=False, chars_as_strings=True
    )
    data = mat_to_pickle_dict(
        mat, n_dof=args.n_dof, transpose=args.transpose, root=args.root
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(data, f)

    n = len(data["labels"])
    dof = data["qp"][0].shape[1]
    print(
        f"已写入 {args.output}：{n} 条轨迹，n_dof={dof}，labels={data['labels']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
