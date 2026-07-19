# Changelog

## [0.4.0] - 2026-07-19
### Features(S2 LoRA 微调 Qwen2.5-VL-3B)
- `s2_train_lora.py`(类平衡 5.9k 样本@224²,r=16 LoRA 37M 参数,标签 token 损失,25min/epoch@4090,
  指令与 `agents/vla_qwen.py` PROMPT 严格一致)、`s2_eval.py`、`make_lora_demo.py`;
  QwenAgent 支持 `lora_path`;反捷径套件支持 `--agent qwen`。适配器 142MB 已 gitignore(>GitHub 100MB)。
### Notes & Caveats
- held-out 生成准确率 0.603(DECLARE 52/52;ROTATE_Y_POS 10/54——与 CNN 同样的"何时停转"盲区)。
- 闭环(10/10/10,seed 123):impossible 10/10 零误报;feasible 4/10;rotation 0/10(左右振荡限环);
  131ms/步。**大模型也没跨过单斜视角感知墙**(BC-CNN 为 5/10, 1/10, 3.6ms)。
- **反捷径对比(核心结果)**:颜色交换 CNN 0.50→**LoRA 0.975**(语言先验消除颜色捷径,靠"物体性"
  而非颜色识别);临界孔 CNN 0.575→LoRA 0.550(**精确空间占据两者都不会**,与几何精度天花板一致);
  基线 0.95 / 尺寸×1.35=0.925 / ×0.70=1.0 / 相机抖动 1.0;所有探针误报 0。
- 四方对比:oracle 30/30 / 零样本 VLA 0/12(碰撞率1.0) / BC-CNN 认输10/10+控制6/20 /
  LoRA-VLA 认输10/10+控制4/20。S1/S2 相对零样本的增益全部来自"学会认输"+一半的平移对齐。

## [0.3.0] - 2026-07-19
### Features(S1 行为克隆 + 反捷径测试)
- 状态式闭环专家 `agents/expert.py`(四元数贪心旋转纠正+实时对齐重算,可重标注任意偏离状态)。
- BC 全链:`s1_collect_bc_data.py`(DAgger-lite 噪声注入采集,14.5k 帧 @224²,存 /data)、
  `s1_train_bc.py`(类别加权 CE + [dx,dz,rot_rem,feasible] 辅助回归)、`s1_eval_bc.py`、
  `s1_dagger.py`、`make_bc_demo.py`;权重 `results/weights/bc_policy.pt`(3.6ms/步)。
- **反捷径套件** `s1_anti_shortcut.py`(规格书第九节):尺寸外推 ±/相机抖动/颜色交换/临界孔六探针,
  GT 全部求解器重验;结果 `results/metrics_anti_shortcut.json`。
### Design Rationale
- 探针只考"执行前可行性判断"(first-action DECLARE),因为这是 BC 唯一稳定学会的能力。
### Notes & Caveats
- 闭环:impossible 10/10(零误报),feasible 5/10,rotation 1/10。瓶颈已定位:专用几何回归网在
  224² 斜视角图上只能做到 dx 6cm/dz 5cm/转角 10°(已收敛),而对齐需要 2.5cm/2.5° ——
  单一固定斜视角的感知精度天花板,非训练技巧问题(DAgger 一轮无效,proprio 反而更差)。
- 反捷径:baseline 1.0 / 尺寸×1.35 0.975 / ×0.70 1.0 / 相机抖动 1.0 / **颜色交换 0.50(捷径,
  impossible 召回 0/20)** / **临界孔 0.575(差 1.5cm 的孔骗过它,召回 3/20)**。
  结论:学到的是"颜色锚定的粗粒度尺寸比较",对尺度/视角迁移,但不是精确空间占据推理。
- 未覆盖(需 Level 2 场景改造):真新形状(L 形/圆柱)、同投影不同深度、同起终点不同中间轨迹。

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

## [0.2.0] - 2026-07-19
### Features
- 零样本 VLA baseline:`agents/vla_qwen.py`(Qwen2.5-VL-3B,动作菜单 prompt→动作词)+
  `scripts/run_vla.py`(与 oracle 同 seed 同关卡同指标)。
### Notes & Caveats
- 结果:12/12 全败,行为 100% 塌缩 MOVE_FORWARD(不对齐/不旋转/不认输),碰撞率 1.0,
  无解关召回 0。与 2D 版(0.07)一致且更彻底——3D 姿态/可行性推理零样本完全不具备。
- 运行环境:需 transformers(habvln,已补装 pybullet);physpass 环境跑 oracle/测试。
