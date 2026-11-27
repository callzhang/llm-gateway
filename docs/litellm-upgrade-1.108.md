# LiteLLM 1.99.0 → 1.108.19 升级评估（2026-11-26）

## 结论

**建议升级，目标 `1.108.19`（1.108 LTS 线），先 GPU4、后 CN。** 但不能只换包：下面的「随升级上线的改动」要和换版本
同时上线。改动已写好，在分支 `feat/litellm-1.108`。**这个分支不能在 1.99.0 上部署**（它去掉了重试记录修复）。
没有部署、没有重启服务、没有改线上环境和线上数据库。

与升级无关、现在就该做的一件事：`ca223b2`（重试记录修复）还没生效。2026-11-26 查到两区 LiteLLM 都没重启过
（GPU4 自 11-23，CN 自 11-17），CN 的目录里还没有这个回调文件。见步骤 0。

升级还会修好一个现网问题：1.99.0 续接 `previous_response_id` 时丢历史（见下一节）。

## `previous_response_id` 续接（1.99.0 的现有缺陷）

上游 [PR 37956](https://github.com/BerriAI/litellm/pull/37956)（2026-08-22 合并）不在 1.99.0 里，在 1.108.19 里
（合并提交是 `v1.108.19` 的祖先）。2026-11-26 实测：真实 proxy + 仓库 `config.yaml` 和回调 + 临时 Postgres +
假上游，每条链 3 轮，检查第 2、3 轮发给上游的内容里有没有之前各轮的用户输入；续接前等 8 秒和不等各一次。

| 请求形式 | 1.99.0 | 1.108.19 |
|---|---|---|
| 非流式，字符串 `input` | 历史完整 | 历史完整 |
| 非流式，列表 `input`（两种写法） | **只剩上一轮回答，用户输入全丢，仍返回 200** | 历史完整 |
| 流式，用 `response.completed` 的 id | **410 `has no stored history`** | 历史完整 |
| 流式，用 `response.created` 的 id | **410 `was not issued by this gateway`** | 历史完整 |

流式两行的 410 是本仓库 `previous_response_hook` 在 LiteLLM 查不到历史时的回答。1.99.0 上的结果与
`retry_breadcrumb_hook` 无关（列表丢失由另一会话在完整代理里单独复现过）。回滚到 1.99.0 会带回这些问题。

## 版本选择

| 版本 | 取舍 |
|---|---|
| **1.108.19** | 选它。LTS 线最新补丁（11-26 发布）；含 PR 39491 修复；下列公告除最后一行外都已修 |
| 1.100.1+ / 1.101.x | 只解决日志问题；GHSA-m8fp 到 1.103.0 才修；LTS 试行期间这两条线不再维护 |
| 1.100.0 | 不能用（重试记录无限增长） |
| 1.109–1.111 | 变更更多；1.111.0 评估时刚发布一天 |

LiteLLM 的 LTS 是 2026-11-02 起为期一个月的试行（[说明](https://docs.litellm.ai/docs/enterprise/lts_pilot)），
实施前确认 1.108 是否仍在维护。

影响 1.99.0、与本网关有关的安全公告：

| 公告 | 内容 | 修复版本 |
|---|---|---|
| GHSA-m8fp-3f3g-j5p7（高） | 请求体自带 `api_key` / `api_base` 会污染其他用户的部署。公告称默认配置受影响；我们的配置下用 `api_key` 没有复现 | 1.103.0 |
| GHSA-7hp6-4w63-5g45（严重） | `internal_user` 提权为管理员并执行命令 | 1.100.4 |
| GHSA-4xm2-x5g5-wc8q（中） | 没有绑定用户的虚拟 key 能读到同类 key 的元数据 | 1.108.13 |
| GHSA-2364 / fjxr / wpm4（中） | 缓存与花费接口越权读取；团队 fallback 越权 | 1.108.0 |
| GHSA-w5h5-479w-p5hx（高） | JWT / SSO 邮箱匹配串号；本网关没配 JWT / SSO | 仅 1.111.0 |

## 随升级上线的改动

| # | 问题（均在临时环境实测） | 改动 |
|---|---|---|
| 1 | 顶层 `priority`：1.99 返回 500；1.108 把 0 原样转给 model_manager，把其他值当成自己的调度参数吃掉 | `request_priority_hook.py` 把它挪进 `extra_body` 再检查 |
| 2 | `/v1/messages`、`/v1/completions` 现在就能让没授权的 key 拿到优先级 0（1.99 已如此） | 同一个 hook：客户端自带的值在所有路由上检查 |
| 3 | 超预算的状态码从 429 变 422（两区各有 3 个 key 设了预算） | 两个 config 加 `budget_exceeded_status_code: 429` |
| 4 | `retry_breadcrumb_hook.py` 会盖掉上游修好的 `log_retry` | 删文件和回调；测试改为 `tests/test_retry_breadcrumbs.py` |
| 5 | 新列 `total_spend` 会被 CN 的 key 同步每天用 GPU4 的值覆盖 | 同步脚本排除列表加 `total_spend` |
| 6 | `test_model_manager_cancellation` 的假日志对象缺字段 | 加 `call_type="aresponses"` |

CN 的 LiteLLM 不加载 `request_priority_hook`（现状如此，本次没改），CN 入口对优先级没有限制。

## `Router.log_retry` 与回调的去留

1.108.19 每次失败记：`model_group`、`deployment_id`、`model`、`base_model`、`exception_type`、`exception_string`、
`attempted_retries`，另在 metadata 记 `request_retry_count`；最多 4 条，只放在本请求的 metadata 里。
**不再复制 `input` / `messages` / 图片**：实测一条经过 order 1 → order 2 的请求，内容在日志里只出现一次。
唯一残留：`exception_string` 是完整的 `str(e)`、不截断，引擎报错若引用请求原文会进日志（只涉及本请求）。

**删除 `retry_breadcrumb_hook.py`。** 保留会替换上游实现，丢掉 `request_retry_count`（请求级重试上限用它）和按次计费用的字段。

## 测试结果

`tests/` 里与 LiteLLM 相关的文件（回调加载、proxy stream、各 hook、context fit、取消、TTS 配置与接入网关）。

| 环境 | 结果 |
|---|---|
| 现网 1.99.0 环境 | 120 通过 |
| 1.108.19，仓库原样 | 118 通过，6 失败 |
| 1.108.19，分支 `feat/litellm-1.108` | 125 通过，0 失败 |
| 1.99.0 环境跑同一分支 | 125 通过，2 失败（重试记录回归测试：1.99 去掉回调后会串内容，符合预期） |

「仓库原样」的 6 个失败：
- `test_upstream_log_retry_...` 2 个子测试：预期失败，上游已修。
- `test_with_the_hook_...` 2 个子测试：断言用的 `Router.previous_models` 属性上游已删除。
- `test_configured_responses_proxy_normal_request_completes`、`..._disconnect_does_not_cancel_peer`：测试替身缺 `call_type`。

新环境另装了 Pillow、transformers、jinja2 才能跑 model_manager 的测试（线上 model_manager 不用这个环境）。

key 同步脚本的改动没有自动化测试（依赖两个 Postgres 和 docker），用临时库手工验证：新→新时 CN 的
`total_spend`、`spend` 保持原值、别的字段照常更新；新→旧同步成功。

另做了新旧版本对比（真实 proxy + 临时 Postgres + 假上游，用仓库当时的 `config.yaml`）。

两版一致：SSE keepalive；伪造 `previous_response_id` 返回 410；`previous_response_id` 跨版本续接（旧发新续、新发旧续，只测了非流式、字符串 `input`）；
order 1 → order 2（503、断连、TTS 503，含流式）；限定模型的 key 走 order 2；客户端断开后取消上游（chat / responses，流式与否）；
TTS 的 `voice` / `instructions` / `response_format` 和 ASR 表单字段的转发（TTS 参数此前只在 1.86.2 验证过）。

有变化、不需改网关，但调用方可能感知：
- Responses 流式事件：message 项现在有 `output_item.added` / `content_part.added`，事件带 `sequence_number`；
  纯工具调用的回答不再多出一个空 message 项。
- 错误体的 `error.type` 有值了，部分 `error.message` 不再带 `Received Model Group=...`。
- 流中途断开时不再向 order 2 重发一次（客户端两版收到的都是错误）。
- chat 流式块带 `finish_reason: null`；`provider_specific_fields.refusal` 没有了。
- 顶层的非标准参数（如 `chat_template_kwargs`）从 500 变为转给上游。

## 1.99 之后的变更中与我们有关的

- **数据库**：新增 63 个迁移（180 → 243 条记录）。用两区线上库的**结构和迁移记录副本（不含数据）**演练，
  均成功、迁移后与 1.108 的 schema 无差异。对现有表以加列为主；`LiteLLM_SpendLogs` 加两列和两个并发建的索引
  （表本体 GPU4 122 MB、CN 254 MB；真实数据上的耗时没测）。
- **没迁移就启动**：1.108 能启动、`/health/liveliness` 正常，但 key 校验全部 500（缺列）。必须先迁移再换版本。
- **Prisma**：新环境要 `prisma generate`；1.104 起默认 v2 迁移器，迁移失败时代理拒绝启动。
- **回滚**：1.99 在迁移后的库上能正常读写。实测 1.99 开着 schema 更新启动没有删任何列（没有待应用迁移），
  但[上游警告](https://docs.litellm.ai/docs/proxy/database_migrations)旧版在迁移恢复路径上会删掉新表新列，
  所以回滚必须保持 `DISABLE_SCHEMA_UPDATE=true`。
- **回调接口**：仓库的回调在两个 config 下都能按路径加载；1.108 起加载不了的日志回调会让代理拒绝启动。
- **不涉及**：库里没有覆盖 config 的设置、没有库内模型、没有 key 级 `router_settings`，所以 1.103 的
  「config 文件优先」和 1.108 的 fallback 权限默认开启不影响现状；主密钥长度够（1.104 起拒绝弱密钥）。
- **环境**：用独立 venv，不在 `llm-gateway-vllm` 里原地升级（它和 vLLM 共用）。回滚只需改回 `GATEWAY_PY_BIN`。

## 实施步骤

命令不会打印连接串。两区的 `gateway.env` 里 `DATABASE_URL` 都没有引号。

### 步骤 0：让现版本的修复先生效（独立于升级）

GPU4：

```bash
systemctl --user restart llm-litellm
```

CN 的目录落后于 HEAD（`previous_response_hook.py`、key 同步脚本也不同），所以这一步只传两个文件
（`config.cn.yaml` 与现网只差回调那两行）：

```bash
git -C /home/derek/Projects/llm-gateway archive ca223b2 retry_breadcrumb_hook.py config.cn.yaml | ssh a100-1 'tar -x -C /data/derek/llm-gateway'
```

```bash
ssh a100-1 'sudo systemctl restart llm-litellm'
```

重启各有几十秒 502。之后的清理见 [spend-log-retry-leak.md](spend-log-retry-leak.md)。

### GPU4

在 `/home/derek/Projects/llm-gateway`。这是线上目录：到第 5 步才合并分支，合并后马上做第 6 步，别让旧版本带着
去掉回调的 config 重启。

1. 建环境（不影响线上）：

```bash
V=/home/derek/venvs/litellm-1.108.19; /home/derek/miniforge3/envs/llm-gateway-vllm/bin/python -m venv $V && PYTHONNOUSERSITE=1 $V/bin/pip install -r .worktrees/litellm-1.108/docs/litellm-1.108.19-requirements.txt && PYTHONNOUSERSITE=1 PATH=$V/bin:$PATH $V/bin/prisma generate --schema $V/lib/python3.12/site-packages/litellm/proxy/schema.prisma
```

2. 备份库（不含 spend log 数据，避免再复制一份含他人内容的日志）：

```bash
mkdir -p -m 700 ~/litellm-backups; url=$(grep '^DATABASE_URL=' gateway.env | cut -d= -f2-); PGURL=${url%%\?*} docker run --rm --memory=1g --network host -e PGURL postgres:16 sh -c 'pg_dump "$PGURL" -Fc --exclude-table-data="\"LiteLLM_SpendLogs\""' > ~/litellm-backups/gpu4-pre-1.108.dump
```

3. `gateway.env` 加一行 `DISABLE_SCHEMA_UPDATE=true`（此后任何版本重启都不自动改库）。
4. 迁移（1.99 可继续服务）。要求退出码 0：

```bash
(export DATABASE_URL="$(grep '^DATABASE_URL=' gateway.env | cut -d= -f2-)" LITELLM_MASTER_KEY="$(grep '^LITELLM_MASTER_KEY=' gateway.env | cut -d= -f2-)"; V=/home/derek/venvs/litellm-1.108.19; PYTHONNOUSERSITE=1 PATH=$V/bin:$PATH DISABLE_SCHEMA_UPDATE=false $V/bin/litellm --skip_server_startup)
```

5. 把 `feat/litellm-1.108` 合并进当前部署分支（`git merge --no-ff feat/litellm-1.108`），在新 venv 里跑一遍上面的测试。
6. `gateway.env` 加 `GATEWAY_PY_BIN=/home/derek/venvs/litellm-1.108.19/bin`，`systemctl --user restart llm-litellm`。
7. 验证：版本号是 1.108.19；chat、流式 responses、TTS、ASR 各一次；`previous_response_id` 续接按下面两条查：
   第 1 轮用列表 `input` 给一个暗号，第 2 轮让模型复述（非流式、流式各一次，流式用 `response.completed` 的 id），
   都要返回 200 且答出暗号；
   日志里没有 `does not exist in the current database`；spend log 有新行；
   `sudo systemctl start llm-litellm-key-sync.service`（在 A100 上）仍显示 `synced keys=...`。

回滚：删掉 `GATEWAY_PY_BIN` 行，**保留 `DISABLE_SCHEMA_UPDATE=true`**，`git revert -m 1` 那个合并提交（把重试记录回调带回来，
否则 1.99 会再次串日志），重启。数据库不回退。

### CN（A100）

GPU4 稳定后再做。**顺序不能反**：CN 先升级会让 key 同步失败（`total_spend` 非空约束，实测）；GPU4 先升级则同步正常。
这一步的建环境命令没有在 A100 上演练过。

1. 传代码前先建环境（依赖清单单独传）：

```bash
scp /home/derek/Projects/llm-gateway/docs/litellm-1.108.19-requirements.txt a100-1:/data/derek/litellm-1.108.19-requirements.txt
```

```bash
ssh a100-1 'V=/data/derek/venv-litellm-1.108.19; /data/derek/bin/uv venv --python /data/derek/uv-python/cpython-3.12.16-linux-x86_64-gnu/bin/python3.12 $V && /data/derek/bin/uv pip install --python $V/bin/python -r /data/derek/litellm-1.108.19-requirements.txt && PYTHONNOUSERSITE=1 PRISMA_BINARY_CACHE_DIR=$V/share/prisma-python PATH=$V/bin:$PATH $V/bin/prisma generate --schema $V/lib/python3.12/site-packages/litellm/proxy/schema.prisma'
```

2. 备份：

```bash
ssh a100-1 'cd /data/derek/llm-gateway && umask 077 && sudo -n docker exec -e PGPASSWORD="$(cut -d= -f2- pg-local.env)" llm-litellm-pg pg_dump -h 127.0.0.1 -p 15433 -U litellm -d litellm -Fc --exclude-table-data="\"LiteLLM_SpendLogs\"" > /data/derek/llm-gateway-backups/cn-pre-1.108.dump'
```

3. 手动迁移（`disable_prisma_schema_update` 和 `DISABLE_SCHEMA_UPDATE=true` 保持不动）：

```bash
ssh a100-1 'cd /data/derek/llm-gateway && V=/data/derek/venv-litellm-1.108.19 && export DATABASE_URL="$(grep "^DATABASE_URL=" gateway.env | cut -d= -f2-)" LITELLM_MASTER_KEY="$(grep "^LITELLM_MASTER_KEY=" gateway.env | cut -d= -f2-)" && PYTHONNOUSERSITE=1 PRISMA_BINARY_CACHE_DIR=$V/share/prisma-python PATH=$V/bin:$PATH DISABLE_SCHEMA_UPDATE=false $V/bin/litellm --skip_server_startup'
```

4. A100 的 `gateway.env`：先抄下原来的 `GATEWAY_PY_BIN`、`PRISMA_BINARY_CACHE_DIR`，改为
   `/data/derek/venv-litellm-1.108.19/bin`、`/data/derek/venv-litellm-1.108.19/share/prisma-python`。
5. 传代码并重启（约 70 秒 502）。整个 HEAD 会一并带去 CN 目录还没有的其他改动，先确认它们可以上线：

```bash
git -C /home/derek/Projects/llm-gateway archive HEAD | ssh a100-1 'tar -x -C /data/derek/llm-gateway'
```

```bash
ssh a100-1 'sudo systemctl restart llm-litellm'
```

6. 验证同 GPU4 第 7 步，再跑 `sudo systemctl start llm-litellm-key-sync.service`。

回滚：`gateway.env` 两个变量改回原值，把 revert 后的代码传过去，`sudo systemctl restart llm-litellm`。
数据库不回退；旧版 `/data/derek/venv-litellm` 在确认稳定前不删。

## 没有验证的

- 真实负载下的内存和 CPU（这条线上有过 LiteLLM 内存问题，升级后看 `llm-litellm-mem-watch`）。
- 真实数据量下的迁移和建索引耗时。
- 真实调用方（编码 agent、memory 服务）对新的 Responses 流式事件的兼容性。
- 真实引擎：对比用的是假上游。
- 1.99.0 写下的列表 `input` 历史，升级后续接能否带回。
- CN 上建环境和下载 Prisma 工具链。

## 名词表

| 名词 | 含义 |
|---|---|
| order 1 / order 2 | 同一模型的本区部署 / 对区部署；order 1 失败才用 order 2 |
| spend log | LiteLLM 的请求日志表 `LiteLLM_SpendLogs`，`previous_response_id` 历史也从这里重建 |
| 重试记录 | `metadata.previous_models`，失败尝试的记录 |
| 回调 / hook | config 的 `callbacks` 里按路径加载的 Python 文件 |
| 迁移 | 升级版本时对数据库结构做的变更 |
| key 同步 | CN 每天从 GPU4 库复制虚拟 key 的定时任务 |
