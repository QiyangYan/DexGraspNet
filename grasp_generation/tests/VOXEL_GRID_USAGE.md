# 体素网格SDF可视化 - 快速指南

## 核心功能

在物体的bounding box内生成均匀的体素网格，计算每个体素点到手部模型的SDF距离，使用颜色区分：
- 🔴 **红色**: 点在手部内部（穿透）
- 🟢 **浅绿色**: 点在手部外部

## 快速开始

```bash
cd /home/guizhewei/guizhewei/DexGraspNet/grasp_generation

# 基础用法：体素网格 + 所有点
python tests/visualize_penetration_sdf.py --num 0 --use_voxel_grid

# 只看穿透区域（红色点）
python tests/visualize_penetration_sdf.py --num 0 --use_voxel_grid --show_inside_only

# 包含手部模型
python tests/visualize_penetration_sdf.py --num 0 --use_voxel_grid --show_hand
```

## 常用参数组合

### 1. 清晰定位穿透区域
```bash
python tests/visualize_penetration_sdf.py --num 0 \
    --use_voxel_grid \
    --show_inside_only \
    --show_hand \
    --point_size 4
```
**效果**: 只显示红色穿透点，配合手部模型，清楚看到穿透发生的位置。

### 2. 完整空间分析
```bash
python tests/visualize_penetration_sdf.py --num 0 \
    --use_voxel_grid \
    --show_hand \
    --show_object_mesh \
    --voxel_resolution 32
```
**效果**: 显示手部、物体网格和所有体素点，全面了解空间占用。

### 3. 高精度分析
```bash
python tests/visualize_penetration_sdf.py --num 0 \
    --use_voxel_grid \
    --voxel_resolution 64 \
    --show_inside_only
```
**效果**: 64³体素网格（262144个点），精细观察穿透区域（需要较长计算时间）。

### 4. 对比内外分布
```bash
# 先看外部点
python tests/visualize_penetration_sdf.py --num 0 --use_voxel_grid --show_outside_only

# 再看内部点
python tests/visualize_penetration_sdf.py --num 0 --use_voxel_grid --show_inside_only
```
**效果**: 分别查看内部和外部点的分布情况。

## 参数调整建议

### 体素分辨率选择

| 分辨率 | 点数 | 用途 | 计算速度 |
|--------|------|------|----------|
| 16³ | 4,096 | 快速预览 | 很快 |
| 32³ | 32,768 | 标准分析（默认） | 快 |
| 64³ | 262,144 | 精细分析 | 较慢 |
| 128³ | 2,097,152 | 超精细（慎用） | 很慢 |

**建议**: 
- 初次查看用 `32`
- 需要细节时用 `64`
- `128` 仅在必要时使用（可能需要数分钟）

### 点大小调整

```bash
# 小点（密集显示）
--point_size 2

# 默认大小
--point_size 3

# 大点（稀疏时推荐）
--point_size 5
```

## 输出信息

运行脚本时会输出：

```
============================================================
可视化Penetration SDF
物体: 47_008_pudding_box
结果索引: 0
设备: cuda
============================================================

使用体素网格模式，分辨率: 32^3 = 32768 个点
物体Bounding Box:
  Min: [-0.0652, -0.0414, -0.0892]
  Max: [0.0652, 0.0414, 0.0892]
  Size: [0.1304, 0.0828, 0.1784]

============================================================
穿透统计信息
============================================================
总采样点数:        32768
穿透点数 (SDF>0):  1234 (3.77%)
外部点数 (SDF<=0): 31534 (96.23%)
============================================================
SDF距离统计:
  最大值 (最大穿透):  0.008234
  最小值:             -0.045123
  平均值:             -0.002345
  标准差:             0.006789
============================================================

可视化: 所有点, 共 32768 个点
正在打开可视化窗口...
```

## 与表面采样点模式的区别

| 特性 | 体素网格模式 | 表面采样点模式 |
|------|--------------|----------------|
| 采样方式 | bbox内均匀网格 | 物体表面FPS采样 |
| 点数量 | 可调节（默认32³） | 固定2000 |
| 颜色 | 二值（红/浅绿） | 连续渐变 |
| 用途 | 空间穿透分析 | E_pen计算验证 |
| 与energy.py一致 | ❌ | ✅ |

## 注意事项

1. **需要CUDA**: TorchSDF库需要CUDA才能计算距离
2. **内存占用**: 高分辨率（64³以上）会占用较多内存
3. **计算时间**: 
   - 32³ 约 1-2秒
   - 64³ 约 10-30秒
   - 128³ 约 2-5分钟
4. **可视化性能**: 点数过多时，Plotly交互会变慢

## 疑难解答

### Q: 显示的点太密集，看不清
**A**: 减小分辨率或只显示穿透点
```bash
--voxel_resolution 16 --show_inside_only
```

### Q: 想看更多细节
**A**: 增加分辨率并放大点
```bash
--voxel_resolution 64 --point_size 4
```

### Q: 计算太慢
**A**: 降低分辨率
```bash
--voxel_resolution 24
```

### Q: 想验证energy.py中的E_pen计算
**A**: 不使用体素网格模式，使用默认的表面采样点模式
```bash
python tests/visualize_penetration_sdf.py --num 0
```

