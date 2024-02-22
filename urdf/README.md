# URDF 学习笔记
待解决问题

1：urdf中如何定义的各连杆的局部坐标系，如何定义关节旋转的正方向，为右手系还是左手系

2：urdf中的inertia属性是针对于哪一坐标系得到的

3：isaacgym中得到的body_state_tensor中的position和rotation与urdf的定义有何联系

4：isaacgym中是否可以实时读取各link的inertia属性，该属性是针对局部坐标系还是世界坐标系，局部坐标系的话该属性应该与link的位姿无关，需验证

## 一：URDF性质
### 1：定义
URDF全称为Unified Robot Description Format，中文可以翻译为“统一机器人描述格式”。URDF是一种基于XML规范、用于描述机器人结构的格式。

### 2：坐标系
URDF中的坐标系分为joint坐标系和inertial坐标系。当前joint坐标系的描述是相对于上一个joint坐标系（即父link的joint坐标系），每个link的inertial坐标系、visual坐标系和collision坐标系均为相对于当前joint坐标系的变换。

因此在URDF建模中，先建立joint之间的坐标系变换关系，再定义joint之间的link的物理属性。唯一例外是base link，base link的物理属性是相对于世界坐标系建立的，base link的joint坐标系即世界坐标系。

### 3：特性
1：URDF是基于xml基础的，因此只能定义树形结构，无法定义存在回环的机器人结构

2：URDF是先定义joint，后定义link


## 二：URDF属性
### 1：joint：关节属性
#### ①：