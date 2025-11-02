## 第12章 课程小实验设计（Lab）

### 开篇段落
本章是理论与实践的桥梁。在前面的章节中，我们系统性地探讨了从模态表征到 Sim-to-Real 的完整技术链条。现在，我们将通过一系列精心设计的小型实验（Lab），将这些抽象概念转化为可触摸、可调试、可度量的代码实践。这些实验的目标并非构建庞大的系统，而是聚焦于 VLA 落地过程中的关键技术点，强调**可复现性、低成本与量化分析**。所有实验都将基于我们课程中使用的 `RLinf` 分布式强化学习框架，让您在解决具体问题的同时，深度体验其灵活的 Worker 架构、配置驱动的工作流以及为性能与安全所做的工程优化。每个实验都配有软件仿真环境，确保在不同硬件条件下均可完成，最终目标是让您具备独立分析、实现并评估 VLA 核心模块的能力。

---

### Lab A：频域平滑的循迹控制（行动模态与舒适度）

*   **实验目标**：深刻理解第 4 章中“行动即信号”的理念。通过在 `RLinf` 的 `RolloutWorker` 中引入一个简单的后处理滤波器，直观感受行动轨迹的平滑性（Jerk/跃度）对系统质量的影响，并学会使用频域分析工具（FFT）来量化“舒适度”。
*   **背景知识**：行动轨迹的高频分量通常对应着不平滑、高跃度的“急动”，这在自动驾驶中意味着糟糕的乘坐体验，在机器人操控中可能导致机械臂震动或任务失败。低通滤波器可以有效抑制这些高频分量。
*   **资源与环境**：
    *   框架：`RLinf`
    *   仿真器：提供一个简化的 2D 车辆循迹仿真环境 (`Simple-Car-Follow-v0`)，该环境通过 `EnvWorker` 启动。
    *   基线代码：一个预训练好的 PPO 策略，能够基本完成循迹任务，但控制指令（如方向盘转角速度）有明显抖动。
    *   配置文件：`configs/lab_a_baseline.yaml`
*   **实验步骤**：
    1.  **运行基线**：首先，直接运行基线策略，`python examples/run_lab.py --config-name=lab_a_baseline`。观察车辆的运动，并记录下参考轨迹与实际轨迹的误差、以及控制指令序列。
    2.  **实现滤波器**：打开 `rlinf/workers/rollout/hf/huggingface_worker.py`，定位到 `predict` 方法。在该方法返回动作 `actions` 之前，实现一个简单的移动平均滤波器（Moving Average Filter）对动作序列进行平滑。
        ```python
        # 伪代码
        window_size = 5
        # self.action_history 是一个存储最近动作的 deque
        self.action_history.append(new_action)
        smoothed_action = np.mean(self.action_history)
        return smoothed_action
        ```
    3.  **参数调优**：修改配置文件 `lab_a_baseline.yaml`，增加一个 `rollout.smoothing_window_size` 参数，并在 `RolloutWorker` 中读取它。尝试不同的窗口大小（如 3, 5, 10），重新运行实验。
    4.  **数据分析**：编写一个简单的 Python 脚本，读取记录下的控制指令序列，计算其三阶差分（近似于跃度/Jerk），并使用 `scipy.fft.fft` 对其进行快速傅里叶变换，绘制频谱图。
*   **评测指标**：
    1.  **循迹误差**：平滑后轨迹与参考轨迹的均方误差（MSE）：$MSE = \frac{1}{T} \sum_{t=1}^{T} || \mathbf{p}_{actual}(t) - \mathbf{p}_{ref}(t) ||_2^2$。
    2.  **跃度范数**：控制指令序列的平均 L1 跃度：$Jerk = \frac{1}{T-3} \sum_{t=1}^{T-3} |a(t+3) - 3a(t+2) + 3a(t+1) - a(t)|$。
    3.  **频谱分析**：对比平滑前后，控制指令频谱图中高频分量的能量占。
*   **提交物**：
    1.  修改后的 `huggingface_worker.py` 文件。
    2.  一份简短的 Markdown 报告，包含：
        *   不同窗口大小下的循迹误差与跃度范数对比表格。
        *   平滑前后的控制指令时域波形图与频域频谱图。
        *   对“平滑度”与“响应速度”之间权衡（trade-off）的简要分析。
*   **加分项**：将移动平均滤波器替换为更高级的巴特沃斯（Butterworth）或高斯（Gaussian）低通滤波器，并分析其性能差异。

---

### Lab B：相机时间戳与延迟补偿（感知-控制闭环同步）

*   **实验目标**：体验在实际部署中，感知（如相机）与控制之间的延迟（Latency）对系统稳定性的致命影响。通过在 `EnvWorker` 中模拟一个可变延迟，并在 `RolloutWorker` 中实现一个简单的前向预测补偿器，来恢复系统性能。
*   **背景知识**：在现实世界中，从图像采集、传输、模型推断到最终执行器响应，整个链条存在不可避免的延迟。如果策略基于一个“过时”的状态 $s_{t-\Delta t}$ 来决策，其产生的动作 $a_t$ 将作用于未来的真实状态 $s_t$，导致不匹配和不稳定。
*   **资源与环境**：
    *   框架：`RLinf`
    *   仿真器：使用 Lab A 的 2D 车辆循迹环境，但这次任务是在一个狭窄通道内保持车辆居中。
    *   基线代码：一个在该任务上训练好的策略，但在引入延迟后会频繁撞墙。
    *   配置文件：`configs/lab_b_latency.yaml`
*   **实验步骤**：
    1.  **复现问题**：在 `rlinf/envs/env_manager.py` 的 `_simulator_worker` 循环中，为 `result_queue.put(result)` 增加一个 `time.sleep(delay)` 来模拟感知延迟。在 `lab_b_latency.yaml` 中设置 `env.sim_delay = 0.1` (100ms)，运行基线，观察到车辆行驶不稳并最终失败。
    2.  **实现补偿器**：在 `RolloutWorker` 的 `predict` 方法中，在得到动作后，不要立即返回。而是基一个简单的匀速运动模型，预测出 $\Delta t$ 之后的车辆状态，并根据预测出的未来状态对动作进行二次修正。
        ```python
        # 伪代码，在 RolloutWorker.predict 中
        current_obs = ...
        action = self.model.predict(current_obs)
        
        # 简单补偿：假设车辆在延迟期间按当前速度和动作继续运动
        dt = self.cfg.env.sim_delay
        v = current_obs['velocity']
        omega = action['steering_rate'] # 假设动作为角速度
        predicted_yaw_change = omega * dt
        predicted_pos_change = v * dt * np.array([cos(current_obs['yaw']), sin(current_obs['yaw'])])
        
        # 基于预测出的未来状态，可以再次调用模型或应用一个修正规则
        # ... (此处为核心实现)
        
        return corrected_action
        ```
    3.  **对比实验**：在有/无延迟补偿的情况下，分别测试不同延迟（如 50ms, 100ms, 200ms）下的任务成功率。
*   **评测指标**：
    1.  **任务成功率**：在 100 次随机测试中，车辆成功通过通道的百分比。
    2.  **平均生存时间**：车辆在撞墙前平均行驶的步数。
*   **提交物**：
    1.  修改后的 `env_manager.py` 和 `huggingface_worker.py`。
    2.  一份简短的报告，包含：
        *   不同延迟下，有/无补偿策略的成功率对比柱状图。
        *   分析为何简单的运动模型补偿能够有效，以及它的局限性在何处。
*   **加分项**：实现一个卡尔曼滤波器（Kalman Filter）来更精确地估计和预测车辆状态，以应对带噪声的观测和更复杂的动力学。

---

### Lab D：域随机化消融（随机化课程与 OOD 鲁棒性）

*   **实验目标**：动手实践 Sim-to-Real 的核心技术——域随机化（Domain Randomization, DR）。通过对仿真环境的物理参数（如摩擦力、质量）和视觉参数（如光照、纹理）进行随机化，训练出一个在“未见过的”测试环中依然表现稳健的策略。
*   **背景知识**：训练于单一、确定性仿真环境的策略往往会“过拟合”到该环境的特定参数上，导致在现实世界（或参数有差异的另一仿真环境）中性能急剧下降。DR 通过在训练时暴露给策略一个分布的仿真实例，迫使其学习对这些变化不敏感的鲁棒特征。
*   **资源与环境**：
    *   框架：`RLinf`
    *   仿真器：ManiSkill3，使用 `PickCube-v1` 任务。
    *   基线代码：一个在默认 ManiSkill 环境中训练好的策略。
    *   配置文件：`configs/lab_d_dr.yaml`
*   **实验步骤**：
    1.  **定义测试域**：创建一个 `test_env_config.json`，在其中设定一组与默认训练环境**不同**的物理和视觉参数（例如，将方块的摩擦系数 `friction` 增大 50%，颜色改为从未见过的紫色）。运行基线策略，验证其在 `test` 域上的失败率很高。
    2.  **实现域随机化**：在 `rlinf/envs/maniskill/maniskill_env.py` 的 `reset` 方法中，读取 `lab_d_dr.yaml` 中的 `domain_randomization` 配置。在每次重置环境时，从指定的均匀分布（Uniform Distribution）中采样新的参数（如 `cube_friction`, `cube_mass`, `light_intensity`）并应用到仿真器中。
        ```python
        # 伪代码，在 ManiskillEnv.reset 中
        dr_params = self.cfg.domain_randomization
        if dr_params.enabled:
            friction = np.random.uniform(*dr_params.cube_friction_range)
            mass = np.random.uniform(*dr_params.cube_mass_range)
            # ... 调用 maniskill API 设置这些参数 ...
        super().reset()
        ```
    3.  **训练 DR 策略**：使用实现了 DR 的环境，重新训练一个策略。
    4.  **消融研究**：分别只开启物理参数随机化、只开启视觉参数随机化、以及两者都开启，训练三个不同的策略。
*   **评测指标**：
    1.  **域内（In-Domain）成功率**：在默认的、无随机化的环境中，策略的抓取成功率。
    2.  **域外（Out-of-Distribution）成功率**：在 `test_env_config.json` 定义的固定 `test` 域中，策略的 Zero-Shot 成功率。
*   **提交物**：
    1.  修改后的 `maniskill_env.py` 和 `lab_d_dr.yaml`。
    2.  一份简短的报告，包含：
        *   一个表格，对比基线策略、仅物理DR、仅视觉DR、完全DR策略在“域内”和“域外”的成功率。
        *   分析为何 DR 能提升泛化性，以及物理和视觉随机化各自贡献了什么。
*   **加分项**：实现一个域随机化的课程（Curriculum），即随着训练的进行，逐渐扩大随机化参数的范围（例如，从 `[0.8, 1.2]` 倍的默认值，慢慢扩大到 `[0.5, 1.5]` 倍）。

---

### Lab F：运行时屏蔽与优雅降级（安全基线）

*   **实验目标**：为基于学习的 VLA 策略构建一个最后的安全防线——运行时屏蔽（Runtime Shield）。当神经网络策略输出一个可能导致碰撞或违反核心约束的危险动作时，安全屏蔽会介入，将其投影到一个预定义的安全动作集合中，实现优雅降级。
*   **背景知识**：即使经过充分训练和测试，深度学习模型也无法保证在所有情况下都 100% 安全。运行时保障（Runtime Assurance, RTA）系统通过一个简单、可验证的“安全层”来监控和修正“高能但不保证安全”的主策略，是部署高风险自主系统的关键。
*   **资源与环境**：
    *   框架：`RLinf`
    *   仿真器：ManiSkill3，使用一个带机械臂和障碍物的 `ReachAvoid-v1` 任务。
    *   基线代码：一个训练好的策略，能完成到达目标的任务，但在某些角落情况下会与障碍物发生碰撞。
    *   配置文件：`configs/lab_f_shield.yaml`
*   **实验步骤**：
    1.  **识别危险场景**：运行基线策略，记录下发生碰撞时的状态（`state`）和策略输出的动作（`action`），分析碰撞原因。
    2.  **实现安全屏蔽**在 `RolloutWorker` 中，在模型输出动作之后，但在发送给 `EnvWorker` 之前，插入一个 `safety_shield` 函数。该函数实现一个简单的、基于几何的安全规则。
        ```python
        # 伪代码，在 RolloutWorker.predict 之后
        def safety_shield(self, state, action):
            # 获取机械臂末端执行器位置和障碍物位置
            end_effector_pos = state['tcp_pos']
            obstacle_pos = state['obstacle_pos']
            obstacle_radius = 0.1 # 假设障碍物是球体
            
            # 预测执行动作后的下一个位置
            # 注意：这里的动力学模型需要简化，例如假设是简单的积分器
            next_pos_prediction = end_effector_pos + action * self.cfg.control_dt

            # 安全规则：如果预测的下一步会进入障碍物的安全半径内
            if np.linalg.norm(next_pos_prediction - obstacle_pos) < obstacle_radius:
                # 介入：将动作修正为“停止
                return np.zeros_like(action)
            else:
                # 安全：允许原始动作
                return action
        
        original_action = self.model.predict(state)
        safe_action = self.safety_shield(state, original_action)
        return safe_action
        ```
    3.  **评估屏蔽效果**：在开启/关闭安全屏蔽的情况下，重新运行策略，并统计碰撞率和任务成功率。
*   **评测指标**：
    1.  **碰撞率**：在 100 次测试中，发生碰撞的百分比。
    2.  **任务成功率**：成功到达目标且未碰撞的百分比。
    3.  **屏蔽介入率**：安全屏蔽被触发的频率。
*   **提交物**：
    1.  修改后的 `huggingface_worker.py`。
    2.  一份简短的报告，包含：
        *   开启/关闭屏蔽后的碰撞率与成功率对比。
        *   分析安全屏蔽为何能有效避免碰撞，并讨论它可能带来的负面影响（例如，过于保守可能导致任务无法完成）。
*   **加分项**：将简单的规则屏蔽替换为一个基于二次规划（Quadratic Programming）的控制屏障函数（Control Barrier Function, CBF）的简化实现。使用 `cvxpy` 库，将安全约束定义为 $h(x) \ge 0$，然后求解一个最小化与原动作差异的 QP 问题：$\min_{\mathbf{u}_{safe}} ||\mathbf{u}_{safe} - \mathbf{u}_{RL}||^2 \quad s.t. \quad \nabla h(x)\mathbf{f}(x,\mathbf{u}_{safe}) + \alpha(h(x)) \ge 0$。

---

### 本章小结

本章提供了一系列阶梯式的动手实验，引导您穿越 VLA 部署的核心挑战区。
-   **行动质量（Lab A, B）**：您学会了从信号处理的视角来塑造和评估行动，并处理了现实世界中不可避免的延迟问题。
-   **对齐与学习（Lab C, E）**：您构建了从语言到行动的最小闭环，并实践了利用传统控制器（教师）指导神经网络（学生）进行残差学习的先进 Sim-to-Real 范式。
-   **鲁棒与安全（Lab D, F）**：您掌握了通过域随机化提升策略泛化能力的关键技术，并为您的智能体构建了最后一道可验证的安全屏障。
通过完成这些实验，您不仅验证了理论知识，更重要的是，积累了在 `RLinf` 框架下进行调试、分析和创新的宝贵工程经验。

### 常见陷阱与错误 (Gotchas)

1.  **环境与依赖**：确保您的 `conda` 环境已正确安装所有依赖，特别是 `ManiSkill3` 及其渲染后端。在无头服务器上运行时，务必确认使用了正确的渲染模式（如 `headless` 或 `server`）。`PYTHONPATH` 未正确设置是导致模块找不到的常见原因。
2.  **数据维度与类型**：在 `RLinf` 的 `Channel` 中传递数据时，要时刻注意 `torch.Tensor` 的 `shape`、`dtype` 和 `device`。一个常见的错误是在 `ActorWorker` (GPU) 和 `RolloutWorker` (可能在 CPU) 之间传递数据时忘记了 `.to(device)`。
3.  **配置驱动的陷阱**：`RLinf` 高度依赖 `yaml` 配置。一个缩进错误或拼写错误可能导致配置无法析，或被静默地忽略，导致程序行为不符合预期。在修改配置后，养成检查 `main` 函数打印出的最终配置的习惯。
4.  **分布式调试**：`Worker` 在独立的进程中运行，`print` 语句的输出可能会被打乱或重定向。学会使用 `RLinf` 的日志系统，并为关键模块设置不同的日志级别。当出现 Worker 僵死时，检查 `Channel` 是否存在死锁（例如，一个 `get` 操作永远等不到 `put`）。
5.  **过拟合域随机化**：在 Lab D 中，如果随机化的范围过窄，或者随机化的参数与测试域的真实差异正交，策略可能依然无法泛化。好的 DR 需要对现实世界的可能变化有准确的先验知识。
6.  **安全屏蔽的副作用**：在 Lab F 中，过于保守的安全屏蔽可能会导致“冻结”行为（Chattering），即策略在安全边界附近反复触发屏蔽，导致无法取得进展。安全屏蔽的设计需要在安全性和任务性能之间取得精妙的平衡
