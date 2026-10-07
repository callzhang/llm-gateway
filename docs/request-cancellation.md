# 请求取消与共享冷启动

## 已确认的缺陷与修复范围

期望：客户端断开后，该次上游 HTTP 推理连接及时关闭，请求计数释放；其他请求和
共享模型启动不受影响。显式 backend 停止及 gateway 关闭仍然有权取消启动等待。

实际：默认 aiohttp HTTP runner 不取消断开的 handler。等待上游响应头或下一段流式
输出时，请求会继续占用计数。直接全局开启取消会把首个请求拥有的冷启动等待一并
取消，另一个等待者可能再次执行启动。

修复：HTTP runner 保持默认取消行为；每次推理代理监测自身连接关闭并取消该次
上游等待（检查间隔 50ms）。不全局取消 ASR/管理 handler，避免共享 CPU worker
的响应错位。backend 持有唯一启动 Task，各请求通过 shield 等待。客户端取消
只取消自己的等待，不能取消共享 Task。启动失败在
backend 内标记失败并按对象身份释放槽位，即使所有等待者都已经离开；完成回调
读取异常，避免无人接收的任务异常。显式 stop / session shutdown 取消并等待启动
Task；若已生成未就绪进程，则 SIGKILL 自有进程组并 reap，随后关闭连接。已就绪
模型继续保留既有跨 gateway 重启的行为。

不改变模型、max_num_seqs、请求参数、路由策略、响应状态或正文，不新增重试。
本次没有部署、重启共享服务或调用真实 GPU 模型。

## LiteLLM Responses 边界：断连取消配置

仅修 model_manager 还不够：它的直接客户端实际是 LiteLLM，而不是最外层的
Responses 调用方。仓库配置没有声明 `general_settings.cancel_on_disconnect`，
本地安装的 LiteLLM1.99.0 在此设置缺省时继续等待模型调用；外层客户端已断开，
LiteLLM 到 model_manager 的连接却仍在，因此后者无法观察原客户端断开。

最小修复是启用 LiteLLM 已有的 `cancel_on_disconnect: true`。不新增中间层、
重试、fallback、模型参数或等待预算；不改 Responses 路由或正常请求的输出。
代理内部将已断开请求标为499并取消该请求所属的 upstream task；不会把499作为
正常连接上的模型回答。该开关作用于 LiteLLM 的共同请求处理器，不仅 Responses。

新增回归 `test_configured_responses_proxy_disconnect_reaches_gateway_upstream`：
使用真实 Starlette Request/已安装的 LiteLLM aresponses 请求处理器，模拟 ASGI
`http.disconnect`；其下游为真实 localhost HTTP→真实 GpuBackend.proxy→受控慢上游。
只有模型路由、认证准备和日志准备受控，没有生成模型或共享服务。修复前在上游
已进入后断开，请求仍pending，按目标断言RED；配置开启后 processor 取消、下游
HTTP关闭、慢上游收到取消、活动计数归零，断言均发生在测试finally清理之前。

- 当前取消控制完整文件14 PASS，4.24s。
- 完整 gateway pytest：219 PASS、59 subtests PASS、35warnings，53.69s，exit0。
  JUnit有219个testcase元素，suite测试计数278包含59个subtests，0failure/error/skip。
- 两个被读取的 LiteLLM proxy源文件均与安装包RECORD SHA256一致；没有修改库。
- 此测试不是完整部署的FastAPI前端/OpenAI响应转换验收，也不是实际GPU abort证明。
  部署时必须核对实际LiteLLM版本、有效general_settings、代理及路由器加载的提交。

独立SPEC/QUALITY审阅无Critical/Important。首轮审阅的非阻断覆盖缺口随后关闭：
新增LiteLLM入口正常完成和peer隔离控制，仍走真实common processor和localhost
HTTP。正常请求完整读取body后才构造typed Responses结果，断言output_text、status、
model、monitor等待退出和计数归零。两个请求先真实进入同一backend（计数2），
断开其中一个后计数1且peer仍pending，释放上游后peer完整返回body、计数0；全部
业务断言都在contextmanager/finally清理之前。模型路由和认证/日志准备受控，
不能把typed结果构造当作实际provider/OpenAI响应转换的验收。

- 重构后的原回归在仅测试进程设置开关false时，仍触发“断开请求继续pending”的
  预期失败断言：1failure、0error；没有改仓库或共享配置。
- 当前取消控制16 PASS，3.29s；完整gateway pytest：221 PASS、59 subtests PASS、
  35warnings，23.46s，exit0。JUnit221 testcase元素，suite计数280包含subtests，
  0failure/error/skip；三个LiteLLM入口控制均存在并通过。
- 原独立reviewer仅复核本次测试diff及完整JUnit，没有重复跑测试；无Critical、
  Important或新增Minor，原normal/peer覆盖缺口关闭。生产代码没有新增改动。

这些测试覆盖等待响应/流建立前的取消传播，不声称完整FastAPI前端、真实认证或
post-call guardrails、整个streaming生命周期及所有后台任务无泄漏均已验收。
未来启用后台polling模式时需单独验证交互。

本修复补齐断连传播，并不证明 #1036 四个首轮600秒TimeoutError由这一缺口导致，
也不保证让原本超时的生成请求成功。排队与生成分段还需同请求证据。共享交付若
获授权，需要同时使 LiteLLM 新配置和 model_manager 取消修复生效；原先仅重启
model_manager 的验收范围不足，不能在未获完整共享发布授权时擅自重启任一服务。

## 本地验证

基线：`08983f375097aa872e976bf1cd76532115b50701`。

- 未修改基线：133 项 unittest 通过。
- 首轮 RED：5 项测试中，两个 HTTP 取消测试超时，两个共享启动断言失败，正常响应
  对照通过。失败分别为上游等待不取消、启动次数 2 而非 1，以及孤立启动未发布失败。
- 初版修复：9 项请求取消、启动和生命周期测试通过；全仓 unittest 142 项通过。
- 独立审阅发现全局取消会破坏共享 ASR 协议，以及 shutdown 遗留半启动进程。
  两个新增测试 RED（协议 drain 超时、未执行 kill），修法已收窄为推理代理取消，
  并显式清理自有未就绪进程。
- 第二次审阅发现 reaping 期间再次启动的竞态：新增实际 Router 领取对照 RED，
  修复为在第一次等待前标记 closing，而非 failed，防止提前释放 GPU 槽位；
  清理任务归 backend 持有，调用方取消不终止清理。移除 shield 的对照测试
  确认清理会被取消，恢复后通过。
- 最终取消及生命周期测试 13 项；完整 unittest 146 项通过，独立复审无
  Critical/Important，`git diff --check` 通过。
- 测试只使用临时 localhost HTTP 服务及受控启动 double，没有创建模型进程。
- 完整基线及完整修复测试都包含既有 TTS 测试诊断/资源告警；不宣称日志无告警。

运行：`.venv/bin/python -m unittest tests.test_model_manager_cancellation -v`，
全量：`.venv/bin/python -m unittest discover -s tests -q`。
隔离虚拟环境补入 LiteLLM 1.99.0 仅用于已有 hook 测试，不修改共享 Python 环境。

### 正常整合后续 master（2026-10-07）

正常合入 `ee17b81ad402d2f211b577179683b982d7229630`，集成提交
`755eab280d91054bd854ddbfd5087e097756305b`。本次取消实现及其测试与原修复
`ece4eebc75abe8616ebdf02563595ba7107d91c5` 的逐文件 diff 为空；新 master 的
context fitting、历史响应读取和 callback loader 修改保留。

- 取消控制仍为 13 项通过；完整 unittest 146 项通过，但 unittest 不包含全部
  pytest 函数，不能以它替代完整测试集合。
- 首次完整 pytest：215 PASS / 3 FAIL。三个失败均在导入 LiteLLM proxy hooks 时
  缺少 `orjson`，还未进入历史响应断言，不是取消修复引起的业务回归。
- 在本 worktree 的忽略 `.venv` 补齐同版本 `litellm[proxy]==1.99.0` 声明依赖，
  不修改共享环境、服务或仓库 runtime。随后完整 `python -m pytest tests -q`
  终态 exit 0：218 PASS，59 个 subtest PASS，35 warnings，23.80 秒。
  JUnit 共 277 testcase，0 failure/error/skip。
- 环境使用 system-site-packages；pip 报告继承包的版本冲突，测试仍全部通过。
  此结果不证明生产依赖一致或日志无告警；共享发布必须另行核对实际环境。
- 仍未 push、发布或重启共享 gateway；本地测试不替代真实 GPU 释放及保险验收。

## 尚未证明

本地代理关闭连接，不单独证明已部署的 vLLM 在同一时刻释放实际 GPU 序列；需要获
授权发布后的端到端观察。历史保险 #993 的 680 秒 handler 时间仍缺排队/生成分段，
不能归因于本次取消缺陷；#994 的小夹具 PASS 也不能替代扩大夹具验收。

独立复审已经完成；本地提交与共享服务发布仍是不同边界。主进程关闭事件循环时
可能终止尚未完成的后台 reap 等待，已发送自有进程组 SIGKILL 并不单独证明线上
GPU 序列与进程树终态，后续发布验收须回读。四项 Memory Connector 目标不会由
这份本地回归报告自动变成完成。
