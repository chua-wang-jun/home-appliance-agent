# data 目录说明

## 统一知识库

`knowledge_base/`

当前所有可被 RAG 读取的知识文件都放在这里，支持：

- `.txt`
- `.pdf`
- `.docx`

包含家庭设备管家资料、空调/除湿机/净水器维护资料，以及新增的 Word/PDF 示例文档。

## 外部业务数据

`external_data.json`

保存按用户和月份组织的设备运行记录，供 `fetch_external_data` 工具读取。该文件不会被 RAG 入库。

## LangGraph 持久化记忆

`../chat_history/data/checkpoints.db`

由 `chat_history` 记忆模块通过 LangGraph `SqliteSaver` 自动创建，保存每个会话线程的 Checkpoint 状态和消息历史。删除该目录会清空网页端历史对话和 Agent 记忆。

`../chat_history/data/sources.json`

保存每次回答命中的知识来源文件，用于历史消息恢复时展示“参考文件”。

## 向量库

- 生效集合：`knowledge_base_agent`
- 生效目录：`data/knowledge_base`
- 入库记录：项目根目录的 `md5.text`
- 支持文件类型：TXT、PDF、DOCX

网页“知识库管理”中的保存、上传和删除操作会自动重建向量索引。

当前使用 `SqliteSaver + sources.json` 实现会话级持久记忆，尚未启用 `InMemoryStore` 或 `SqliteStore` 跨会话长期语义记忆。
