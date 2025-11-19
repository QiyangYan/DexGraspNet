# Penetration SDF 可视化工具

## 功能说明

`visualize_penetration_sdf.py` 是一个专门用于可视化抓取优化过程中物体表面点penetration SDF的工具。

该脚本支持两种模式：

### 模式1: 表面采样点模式（默认）
- 获取物体表面采样点（与 `energy.py` 中 `E_pen` 计算使用的相同数据）
- 计算每个采样点到手部模型的SDF距离
- 使用连续颜色映射显示SDF值

### 模式2: 体素网格模式（推荐用于分析穿透）
- 在物体bounding box内生成均匀的体素网格
- 计算每个体素点到手部模型的SDF距离
- 使用二值颜色区分：**红色=内部（穿透）**，**浅绿色=外部**
- 可选择只显示内部点或外部点

## 使用方法

### 基本用法

```bash
cd /home/guizhewei/guizhewei/DexGraspNet/grasp_generation

# 表面采样点模式（默认）
python tests/visualize_penetration_sdf.py --object_code 47_008_pudding_box --num 0

# 体素网格模式（推荐）
python tests/visualize_penetration_sdf.py --object_code 47_008_pudding_box --num 0 --use_voxel_grid
```

### 体素网格模式完整参数

```bash
python tests/visualize_penetration_sdf.py \
    --object_code 47_008_pudding_box \
    --num 0 \
    --result_path /home/guizhewei/guizhewei/DexGraspNet/data/experiments/1103/results \
    --use_voxel_grid \
    --voxel_resolution 32 \
    --show_hand \
    --show_object_mesh \
    --point_size 3
```

### 参数说明

**基本参数：**
- `--object_code`: 物体代码（默认: `47_008_pudding_box`）
- `--num`: 结果索引（默认: `0`）
- `--result_path`: 结果文件路径（默认: `/home/guizhewei/guizhewei/DexGraspNet/data/experiments/1103/results`）

**模式选择：**
- `--use_voxel_grid`: 启用体素网格模式（默认: False，使用表面采样点）
- `--voxel_resolution`: 体素网格分辨率，每个维度的点数（默认: 32，即32³=32768个点）

**显示选项：**
- `--show_hand`: 是否显示手部模型（默认: False）
- `--show_object_mesh`: 是否显示物体网格（默认: False）
- `--show_inside_only`: 只显示内部点/穿透点（默认: False）
- `--show_outside_only`: 只显示外部点（默认: False）

**可视化调整：**
- `--point_size`: 点的大小（默认: 3）
- `--sdf_min`: SDF颜色条最小值（仅表面采样模式，默认: -0.01）
- `--sdf_max`: SDF颜色条最大值（仅表面采样模式，默认: 0.01）

## 可视化说明

### 颜色编码

**体素网格模式（二值颜色）：**
- 🔴 **红色**: SDF > 0（点在手部内部，发生穿透）
- 🟢 **浅绿色**: SDF ≤ 0（点在手部外部，无穿透）

**表面采样点模式（连续颜色映射 RdYlGn_r）：**
- 🔴 **红色**: 正值（点在手部内部，发生穿透）
- 🟡 **黄色**: 接近零（点在手部表面附近）
- 🟢 **绿色**: 负值（点在手部外部，无穿透）

### 统计信息

运行脚本后，终端会输出：

```
==============================================================
穿透统计信息
==============================================================
总采样点数:        2000
穿透点数 (SDF>0):  150 (7.50%)
外部点数 (SDF<=0): 1850 (92.50%)
==============================================================
SDF距离统计:
  最大值 (最大穿透):  0.008234
  最小值:             -0.045123
  平均值:             -0.002345
  标准差:             0.006789
==============================================================

E_pen (优化中的穿透能量): -1.234567
注意: E_pen = -sum(distances[distances>0])
计算验证: -sum(distances[distances>0]) = -1.234567
==============================================================
```

### 交互功能

在可视化窗口中：
- 鼠标拖动旋转视角
- 滚轮缩放
- 鼠标悬停在点上显示详细信息（坐标和SDF值）

## 注意事项

1. **需要CUDA**: TorchSDF库需要CUDA才能计算距离，请确保在有CUDA的环境中运行
2. **默认只显示采样点**: 为了清晰显示penetration情况，默认不显示手部和物体网格。如需显示，添加 `--show_hand` 和 `--show_object_mesh` 参数
3. **颜色条范围**: 可以通过 `--sdf_min` 和 `--sdf_max` 调整颜色条范围以获得更好的可视化效果

## 与原可视化工具的区别

- **原工具** (`visualize_result_optimize_omnihand.py`): 全面展示抓取结果，包括手部、物体、接触点等
- **本工具** (`visualize_penetration_sdf.py`): 专注于penetration分析，详细展示物体表面点的SDF分布

两个工具互补使用，可以获得更全面的抓取质量分析。

## 示例场景

### 场景1: 体素网格 - 可视化所有点（推荐）
```bash
python tests/visualize_penetration_sdf.py --num 0 --use_voxel_grid
```
在bbox内生成体素网格，红色显示穿透点，浅绿色显示外部点。

### 场景2: 体素网格 - 只看穿透区域
```bash
python tests/visualize_penetration_sdf.py --num 0 --use_voxel_grid --show_inside_only
```
只显示发生穿透的体素点（红色），清晰定位穿透区域。

### 场景3: 体素网格 + 手部模型
```bash
python tests/visualize_penetration_sdf.py --num 0 --use_voxel_grid --show_hand
```
同时显示手部和体素点，直观看到手部与物体的空间关系。

### 场景4: 高分辨率体素网格
```bash
python tests/visualize_penetration_sdf.py --num 0 --use_voxel_grid --voxel_resolution 64
```
使用64³=262144个点，更精细地观察穿透分布（计算较慢）。

### 场景5: 完整场景分析
```bash
python tests/visualize_penetration_sdf.py --num 0 --use_voxel_grid --show_hand --show_object_mesh
```
显示所有元素：手部、物体网格、体素点，全面分析。

### 场景6: 表面采样点模式（用于E_pen验证）
```bash
python tests/visualize_penetration_sdf.py --num 0
```
使用与energy.py相同的表面采样点，验证E_pen计算。

