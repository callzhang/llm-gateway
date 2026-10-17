# 请求优先级闸门（设计，未实现）

## 问题

期望：生产请求先于 debug 请求，debug 请求先于评测请求；高优先级的请求不因低优先级请求占满模型而排队。

实际：qwen3.8-27b 的 vLLM 以 `--max-num-seqs 8` 运行，调度策略是先到先服务。memory-connector 的 live 后端和 CI 后端（评测）是两个数据库、两个任务队列，互相看不到对方，它们对同一个模型的争用只能在 gateway 这一层仲裁。CI 批次占满 8 个并行序列时，live 的请求在 vLLM 内部排在后面，gateway 不知道、也管不了顺序。

## 范围

只改 chat 类请求（`request_kind == "chat"`，包含 LiteLLM 由 Responses 转成的 chat/completions）。语音、转写、嵌入不变。不改 vLLM 启动参数，不重启 vLLM。

## 方案

在 `GpuBackend.proxy()` 里、`_active_requests` 计数之后、`_forward()` 之前，加一个按 `priority` 排序的闸门。每个 backend（一个 vLLM 实例）一个闸门，容量等于它的 `max_num_seqs`（`ModelConfig.max_num_seqs`，启动时已校验与 vLLM 命令行一致）。放在 backend 里是为了保持 `x-task-id` 的粘性路由：任务已经选定了 slot，只在该 slot 内排队。

- 在途请求数小于容量：直接放行。
- 已满：请求入队，按 `(priority, 到达序号)` 升序出队，数值小的先出。严格优先级，不老化：高优先级一直有请求时，低优先级可能一直排不到（产品上的取舍，由使用方确认）。
- 请求完成、失败或客户端断开时释放名额，并唤醒队首。等待中的请求被取消（客户端断开，已有 `_cancel_forward_on_disconnect` 机制）时从队列移除，不占名额。
- vLLM 内部因此不再排队：进入 vLLM 的请求数始终不超过 `max_num_seqs`。
- 排队中的请求仍计入 `_active_requests`，这样 `_pick` 的最少连接选择和空闲看门狗行为与现在一致。

## 优先级怎么传到 model_manager

LiteLLM 会丢弃客户端 header（见 `_extract_task_id` 的注释），所以只能用请求体字段。

- model_manager 读取请求体顶层的整数 `priority`，取值 0 到 255，数值越小越先调度（与 vLLM、LiteLLM 的约定一致）。
- 缺省（请求没有该字段）按 100 处理：这是定义好的缺省值，不是失败后的兜底。非整数或超出范围返回 400，不静默修正。
- 转发给 vLLM 之前从请求体里删除 `priority`，因为 vLLM 当前调度策略是 fcfs，字段无意义，保持请求体与现状一致。
- 客户端怎么把字段送到这里：LiteLLM 对 Responses API 的顶层 `priority` 返回 500（`unexpected keyword argument 'priority'`，2026-10-17 实测）；嵌套的 `extra_body.priority` 返回 200，并被合并成下游 chat 请求的顶层字段。所以 memory-connector 的 provider profile 要写成 `extra_body: {extra_body: {priority: N}}`（OpenAI SDK 会把外层 `extra_body` 展开成顶层字段，内层才会作为嵌套字段到达 LiteLLM）。实现时需要在 gateway 侧用同样的小请求再验证一次。

约定的取值：生产 0，debug 100，评测 200（只是使用方的配置，gateway 不认识这些名字）。

## 为什么不用别的办法

- LiteLLM 自带的 `scheduler`（`[BETA] Request Prioritization`）：只覆盖 `/chat/completions` 与 `/completions`，我们走 Responses；只有在 `_async_get_healthy_deployments` 返回空（deployment 被限流或冷却）时才让请求排队，而这个 gateway 的 LiteLLM 没有设 `max_parallel_requests`/RPM，所以不会触发；队列是进程内内存堆，没有接 Redis。
- LiteLLM v1.77.3 的 Priority Quota Reservation：按 key 预留 TPM/RPM 比例，是配额，不是排队，超额直接拒绝。
- vLLM `--scheduling-policy priority`：需要重启 vLLM，影响所有调用方；且多副本各自生效。已被明确排除。
- memory-connector 内的 job 队列：live 与 CI 各自一套，彼此不可见，无法仲裁（见 memory-connector 的 `config-driven-channel-concurrency` change）。

## 可观察性

`/admin/status` 的每个 backend 增加 `gate`：`capacity`、`in_flight`、`waiting`（按优先级的计数）。排队的请求在放行时打一行日志：模型、slot、priority、等待毫秒数。

## 测试（确定性，不起 GPU）

- 容量未满时不排队；已满时按 priority 升序、同优先级按到达顺序放行。
- 低优先级先到、高优先级后到：高优先级先放行（严格，不老化）。
- 排队中的请求被取消后从队列移除，不占名额，不阻塞后面的请求。
- 请求完成、上游 5xx、客户端断开都会释放名额（包括 `BackendGoneError` 重试路径只占一个名额）。
- `priority` 缺省为 100；非整数、负数、大于 255 返回 400；转发给上游的请求体不含 `priority`。
- 沿用 `request-cancellation.md` 里的真实 localhost HTTP → `GpuBackend.proxy` → 受控慢上游的测试方式，验证断开时闸门名额被释放。

## 上线与回滚

- 部署需要重启 model_manager；已就绪的模型会跨重启保留（见 `request-cancellation.md`），但重启会中断在途请求，需要选窗口，并由使用方批准。
- 先只在 gateway 上线闸门而不改任何客户端：没有 `priority` 的请求全部按 100，行为等价于一个容量为 `max_num_seqs` 的先到先服务闸门。
- 再让 memory-connector 的 CI profile 带 200、live profile 带 0，用 `/admin/status` 的 `waiting` 与 vLLM `/metrics` 的 `num_requests_waiting` 核对：CI 批次期间 live 请求的等待时间。
- 回滚：回退这次部署，客户端多发的 `priority` 字段在 LiteLLM 层无害（嵌套 `extra_body` 被接受）。

## 未决

- 多副本（scale-out）时每个 backend 一个闸门，不同副本之间不做全局排序；sticky 路由已经把同一任务固定在一个 slot。是否需要模型级全局闸门，等观察到多副本下的不公平再定。
- `priority` 缺省值 100 是否应由配置声明，等第一个非 memory-connector 的调用方出现再定。
