## 消融实验列表

### Fusion Operation & Fusion Stages
- 两种FO：MSDeform(add) & Simple Sampling(cat&map)
- 固定总stage数(n+m)为6
- 四种FS：n=1,2,3,4

#### 实验
- MSDeform, n=1, lr=0.001, score_thr=0.1
- MSDeform, n=2, lr=0.0005, score_thr=0.02
- MSDeform, n=3, lr=0.0002, score_thr=0.02
- MSDeform, n=4, lr=0.0001, score_thr=0.02
- SS, n=1, lr=0.001
- SS, n=2, lr=0.0005
- SS, n=3, lr=0.0002
- SS, n=4, lr=0.0001


### Fusion Locations
- 两种FL：residual block前/后

#### 实验
- residual block前
- residual block后

### 分割辅助头
- 包括分割辅助损失与自注意加权

#### 实验
- 无辅助损失，无加权
- 有辅助损失，无加权
- 有辅助损失，有加权

### 使用的图像特征层数
探究多尺度特征的影响

#### 实验
- MS+1/2/3/4/5层
- SF+1/2/3/4/5层

### 通用性
- VoxelNeXt, PointPillar

#### 实验
- VoxelNeXt+VoD
- VoxelNeXt+TJ4D
- PointPillar+VoD
- PointPillar+TJ4D

### 可选：待验证有效性
#### 实验
- +Sep. Radar Branch
- +Score Map
- +SRB+SM

## 对比实验列表
- PointPillar和VoxelNeXt基线模型
- FocalConv (证明其使用到的图像特征太少，没有分割辅助)
- PointAugmenting (不需要分开处理图像点云特征，没有分割辅助)