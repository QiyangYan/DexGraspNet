# 可视化工具使用说明

## visualize_result_optimize_omnihand.py

原始的可视化脚本，现已添加 `--object_surface_points` 选项来显示物体表面采样点的SDF。

### 基本用法

```bash
# 标准可视化（手部+物体）
python tests/visualize_result_optimize_omnihand.py --num 0

# 添加物体表面采样点SDF可视化
python tests/visualize_result_optimize_omnihand.py --num 0 --object_surface_points
```

### 完整参数

```bash
python tests/visualize_result_optimize_omnihand.py \
    --object_code 47_008_pudding_box \
    --num 0 \
    --result_path /path/to/results \
    --object_surface_points \
    --sdf_point_size 2 \
    --sdf_min -0.01 \
    --sdf_max 0.01
```

### 参数说明

**原有参数：**
- `--object_code`: 物体代码（默认: `47_008_pudding_box`）
- `--num`: 结果索引（默认: `0`）
- `--result_path`: 结果文件路径

**新增参数：**
- `--object_surface_points`: 启用物体表面采样点SDF可视化（需要CUDA）
- `--sdf_point_size`: SDF采样点大小（默认: `2`）
- `--sdf_min`: SDF颜色条最小值（默认: `-0.01`）
- `--sdf_max`: SDF颜色条最大值（默认: `0.01`）

### 注意事项

1. **CUDA要求**: `--object_surface_points` 功能需要CUDA支持
   - 如果当前环境是CPU，脚本会自动在CUDA上临时创建模型进行计算
   - 建议在有CUDA的环境中运行以获得更好的性能

2. **可视化内容**:
   - 默认显示：手部模型（初始+优化后）、物体网格、能量信息
   - 添加 `--object_surface_points` 后：额外显示2000个表面采样点，用颜色表示SDF值
     - 🔴 红色：穿透（SDF > 0）
     - 🟢 绿色：外部（SDF ≤ 0）

3. **统计信息**: 启用 `--object_surface_points` 后，终端会打印详细的SDF统计信息

### 示例

#### 示例1: 普通可视化
```bash
python tests/visualize_result_optimize_omnihand.py --num 0
```
显示手部和物体的标准可视化。

#### 示例2: 添加SDF可视化
```bash
python tests/visualize_result_optimize_omnihand.py --num 0 --object_surface_points
```
在标准可视化基础上，额外显示物体表面采样点的SDF分布。

#### 示例3: 自定义SDF参数
```bash
python tests/visualize_result_optimize_omnihand.py --num 0 \
    --object_surface_points \
    --sdf_point_size 3 \
    --sdf_min -0.005 \
    --sdf_max 0.005
```
使用更大的点和更小的颜色范围，更清晰地观察微小的穿透。

---

## 与 visualize_penetration_sdf.py 的对比

| 特性 | visualize_result_optimize_omnihand.py | visualize_penetration_sdf.py |
|------|--------------------------------------|------------------------------|
| 主要用途 | 全面展示抓取结果 | 专注于penetration分析 |
| 手部显示 | 总是显示 | 可选（`--show_hand`） |
| 物体显示 | 总是显示 | 可选（`--show_object_mesh`） |
| SDF显示 | 可选（`--object_surface_points`） | 总是显示 |
| 能量信息 | 显示在图表上 | 详细打印到终端 |
| 推荐场景 | 日常查看抓取结果 | 深入分析penetration问题 |

**建议**：
- 日常使用：`visualize_result_optimize_omnihand.py`
- 需要分析penetration时：添加 `--object_surface_points`
- 深入研究penetration：使用 `visualize_penetration_sdf.py`

