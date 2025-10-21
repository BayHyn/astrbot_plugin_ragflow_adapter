# RAGFlow Adapter for AstrBot

这是一个为 [AstrBot](https://github.com/Soulter/AstrBot) 设计的插件，旨在集成 [RAGFlow](https://github.com/infiniflow/ragflow) 的检索能力，为语言模型提供动态、实时的上下文增强。

## 功能

- **自动化 RAG**：通过 AstrBot 的 `on_llm_request` 钩子，自动拦截用户的提问，并将其发送到 RAGFlow 进行检索。
- **查询重写 (可选)**：在检索前，可配置使用一个指定的 LLM Provider 对用户原始问题进行优化，使其更适合作为检索引擎的输入。
- **灵活的上下文注入**：支持三种方式将检索到的内容注入到 LLM 的上下文中：
    1.  `system_prompt` (默认): 将检索内容作为系统提示词的一部分。
    2.  `user_prompt`: 将检索内容附加在用户问题的最前面。
    3.  `insert_system_prompt`: 将检索内容作为一条独立的 `system` 消息插入到对话历史中。
- **高度可配置**：所有关键参数，如 RAGFlow API 地址、API Key、知识库 ID、查询重写等，均可在插件配置页面进行设置。

## 实现思路

本插件的核心设计思想是“无感增强”。它不提供任何需要用户手动调用的命令，而是作为一个中间件，在后台自动为所有与 LLM 的交互提供 RAG 能力。

1.  **拦截请求**：利用 `@filter.on_llm_request()` 装饰器，插件可以在每次 LLM 请求发生前执行代码。
2.  **查询与检索**：
    - 插件首先会检查是否启用了“查询重写”。如果启用，它会调用一个 LLM 对用户的原始问题进行改写。
    - 接着，使用最终的查询问题，通过 `httpx` 异步调用 RAGFlow 的 `/api/v1/retrieval` 接口。
3.  **解析与提取**：插件会解析 RAGFlow 返回的 JSON 数据，并只提取其中 `data.chunks` 数组里每个对象的 `content` 字段，确保注入的上下文干净、有效。
4.  **动态注入**：根据用户在配置中选择的 `rag_injection_method`，插件将拼接好的检索内容动态地修改到即将发送给 LLM 的 `ProviderRequest` 对象中。
5.  **继续流程**：插件执行完毕后，AstrBot 框架会将这个被“增强”过的请求对象发送给原始的 LLM Provider，从而实现基于检索的增强生成。

## 配置项说明

| 配置项                       | 类型   | 描述                                                                                   | 默认值                               |
| ---------------------------- | ------ | -------------------------------------------------------------------------------------- | ------------------------------------ |
| `ragflow_base_url`           | string | RAGFlow 服务的 API 地址。                                                              | `http://127.0.0.1:8000/`             |
| `ragflow_api_key`            | string | RAGFlow 的 API Key。                                                                   | (空)                                 |
| `ragflow_kb_ids`             | list   | 要查询的 RAGFlow 知识库 ID 列表。                                                      | `[]`                                 |
| `enable_query_rewrite`       | bool   | 是否启用查询重写以优化用户输入。                                                       | `false`                              |
| `query_rewrite_provider_key` | string | 用于查询重写的 LLM Provider (通过下拉菜单选择)。                                       | (空)                                 |
| `rag_injection_method`       | string | RAG 检索内容的注入方式 (`user_prompt`, `system_prompt`, `insert_system_prompt`)。 | `system_prompt`                      |
