# VoiceTrail

**面向长期语音记忆的事件条件化情感轨迹系统**

[English](README.md) · [架构设计](docs/ARCHITECTURE.md) · [配置说明](docs/CONFIGURATION.md) · [数据格式](docs/DATA_FORMAT.md)

VoiceTrail 面向跨会话语音交互中的长期记忆问题，将语音内容组织为具有时间、事件和主体归属的事实与情感记忆。在问答阶段，系统结合语义检索与事件条件化情感轨迹，从历史会话中选取相关证据，生成有来源支持的回答。

## 核心方法

VoiceTrail 采用从语音理解到记忆问答的端到端流水线：

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

- **事实与情感联合记忆**：将对话事实和情绪观测关联到原始话轮，保留说话人、会话及时间信息。
- **事件条件化情感轨迹**：围绕事件组织跨会话情感变化，区分用户自身的感受与对他人状态的描述。
- **证据协调与动态预算**：根据问题类型路由、筛选和组合事实、情感与原始会话证据，为 Reader 构建紧凑的历史上下文。

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

测试覆盖流水线调度、任务回执、来源与角色约束、数据解析及路径校验。

## License

[Apache License 2.0](LICENSE)。
