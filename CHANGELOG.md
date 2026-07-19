# Changelog

## [0.1.0] - 2026-07-19
### Features(MVP,规格书第十六节)
- PyBullet+EGL 无头 3D 实验台:绿长方体穿红墙黄孔;14 离散动作(6平移+6旋转+STAY+**DECLARE_IMPOSSIBLE**)。
- 离散+连续碰撞:签名间隙、8-substep 扫掠(lerp+slerp)、TTC;实测抓住 tunneling 与"孔内旋转撞框"。
- 可行性 GT 求解器:解析下界筛 + 三面族×5°网格扫掠过孔验证 → 三类标签 + 专家动作序列。
- 按构造生成器(30/50/20)+求解器验证;起点/孔心贴动作网格。
- Gymnasium env(info 按规格)、JSONL 日志、离线指标(CSV+JSON)、四视图渲染+HUD 演示 GIF。
### Design Rationale
- **PyBullet 替代 Unity**:本机无 Unity/Godot、纯终端工作流;PyBullet 原生满足 Python API/无头批量/多模态导出。
- **墙=4凸box**:避免凹网格,凸-凸 GJK 快而精确;**kinematic+performCollisionDetection**:MVP 无动力学,绝不 stepSimulation。
- **可行性标签接真实碰撞引擎**(不只解析几何):impossible 由投影宽度下界保证,可行由扫掠直穿证明;专家计划即 BC 演示数据。
- **相机策略**:固定斜上方主视角(喂模型)+三正交标签视角(不喂模型)——先固定视角验证物理能力,后续随机化排除模板记忆。
### Notes & Caveats
- EGL 插件不支持真·正交投影 → 辅助视图用远距(20m)窄FOV(5.2°)透视近似。
- PyBullet 碰撞 margin ~1e-2:孔内几何间隙 0.05 实测 0.04;成功判定用 clearance>0 已涵盖。
- 求解器 MVP 版只搜"旋转到位→直穿"策略(单墙场景够用且精确);规格书 case 4/7(中途旋转、全局不可行)留待 Level 3+ 用完整 SE(3) A*。
- 实测:oracle 30 episodes 全 PASS(可行 26/26 穿过、无解 4/4 正确 DECLARE),10 秒。
