# VoiceTrail

**面向长期语音记忆的事件条件化情感轨迹系统**

[English](README.md) · [架构设计](docs/ARCHITECTURE.md) · [流式查询](docs/STREAMING_QUERY.md) · [配置说明](docs/CONFIGURATION.md) · [数据格式](docs/DATA_FORMAT.md)

VoiceTrail 面向跨会话语音交互中的长期记忆问题，将语音内容组织为具有时间、事件和主体归属的事实与情感记忆。在问答阶段，系统结合语义检索与事件条件化情感轨迹，从历史会话中选取相关证据，生成有来源支持的回答。

## 核心方法

VoiceTrail 通过分阶段流水线连接语音理解、记忆构建与证据驱动的问答：

```text
语音会话
   │
   ▼
ASR：语音转写与说话人关联
   │
   ▼
Writer：事实记忆与情感观测提取
   │
   ▼
Indexer：E5 语义向量索引
   │
   ▼
VoiceTrail Retrieval
   ├── Schema-aware Routing
   ├── Event-Conditioned Emotion Trail
   └── Dynamic Evidence Budget
   │
   ▼
Reader：基于历史证据的跨会话问答
```

系统围绕三个关键环节展开：

- **来源约束的事实与情感记忆**：绑定原始话轮与引文，保留说话人、断言类别、时间范围及有来源支撑的事件键。
- **事件条件化情感轨迹**：跨会话扩展仅采用非空且精确相等的事件 ID，在候选回填前恢复同一事件遗漏的会话阶段。
- **自适应联合证据**：通过 E5/Schema 路由、来源去重和按问题分配预算，协调原始对话、事实与情感证据。

## 语音增量查询与历史时点检索

[流式查询接口](docs/STREAMING_QUERY.md) 串联 Listening、提前候选准备（Anticipation）、Speech Tail 与最终 Searching。系统可根据部分语音提前准备证据，并使用完整问题确定最终检索上下文。Qwen2.5-Omni HTTP 接口处理完整的短 WAV 分块；原生逐词流式转录需接入相应的 ASR 服务。

QA 支持通过 ISO 日期格式的 `cutoff_record_date` 限定历史记忆范围。Reader 分开统计 `retrieval_latency_ms`、`query_embedding_ms` 和答案生成时间。Writer 基于原始话轮生成用户隔离的稳定事件键，检索时以精确相等的事件 ID 组织跨会话情感轨迹。

## 流水线调度

VoiceTrail 将 ASR、Writer、Indexer 和 Reader 作为独立阶段。调度器通过有界 `asyncio.Queue` 支持阶段间并行执行和背压控制；Writer 使用 SQLite 持久化任务队列管理租约、重试和提交状态。不同阶段可以配置独立的工作线程数量，并保存可追溯的处理回执。

## 项目结构

| 模块 | 代码位置 | 功能 |
| --- | --- | --- |
| Scheduler | `src/voicetrail/scheduler.py` | 异步流水线调度、任务协调 |
| ASR | `src/voicetrail/adapters/asr.py` | 音频转写 |
| Writer | `src/voicetrail/adapters/writer.py` | 事实与情感记忆提取 |
| Validation | `src/voicetrail/adapters/validation.py` | 来源与情感主体校验 |
| Indexer | `src/voicetrail/adapters/indexing.py` | E5 语义向量构建 |
| Retrieval | `src/voicetrail/retrieval/` | 语义路由、轨迹选择与证据预算 |
| Reader | `src/voicetrail/adapters/reader.py` | 基于证据的回答生成 |
| Storage | `src/voicetrail/storage/` | Writer 任务持久化 |
| Runtime | `src/voicetrail/config.py`、`src/voicetrail/cli.py` | 参数配置与命令行入口 |

## 安装

推荐 Python 3.10 及以上版本。

```bash
python -m venv .venv
pip install -e ".[live,test]"
```

复制 `.env.example` 为 `.env`，配置 ASR、Writer、Reader 服务及 E5 模型。完整参数见 [配置说明](docs/CONFIGURATION.md)。

## 使用

VoiceTrail 使用统一的语音会话与 QA 数据格式，适配 **Voxemobench**。输入结构详见 [数据格式](docs/DATA_FORMAT.md)。

检查数据结构：

```bash
voicetrail-run --dataset /path/to/dataset --preflight
```

运行流水线：

```bash
voicetrail-run --dataset /path/to/dataset --output /path/to/output
```

可通过 `--asr-workers`、`--writer-workers`、`--index-workers`、`--reader-workers` 和 `--queue-capacity` 调整调度并发，通过 `--users`、`--sessions`、`--question-limit` 选择处理范围。

## 测试

```bash
python -m pytest -q
```

测试覆盖流水线调度、任务回执、来源与角色约束、事件身份和情感类型、跨会话轨迹恢复、问题时间截止、增量查询及数据校验。

## License

[Apache License 2.0](LICENSE)。
