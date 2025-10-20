# RAGFlow API - 检索知识块

`POST /api/v1/retrieval`

从指定的数据集检索知识块。

---

### 请求

- **方法**: `POST`
- **URL**: `/api/v1/retrieval`
- **请求头**:
  - `Content-Type: application/json`
  - `Authorization: Bearer <您的 API 密钥>`

---

### 请求参数

- **`question`** (string, 必需): 用户查询或查询关键词。
- **`dataset_ids`** (list[string]): 要搜索的数据集的 ID 列表。
- **`document_ids`** (list[string]): 要搜索的文档的 ID 列表。
- **`page`** (integer, 可选): 显示知识块的页面，默认为 `1`。
- **`page_size`** (integer, 可选): 每页的最大知识块数，默认为 `30`。
- **`similarity_threshold`** (float, 可选): 最小相似度得分，默认为 `0.2`。
- **`vector_similarity_weight`** (float, 可选): 向量余弦相似度的权重，默认为 `0.3`。
- **`top_k`** (integer, 可选): 参与向量余弦计算的知识块数量，默认为 `1024`。
- **`rerank_id`** (string, 可选): 重排模型的 ID。
- **`keyword`** (boolean, 可选): 是否启用基于关键词的匹配，默认为 `false`。
- **`highlight`** (boolean, 可选): 是否在结果中启用匹配术语的高亮显示，默认为 `false`。
- **`cross_languages`** (list[string], 可选): 为了实现跨语言关键词检索，应翻译成的语言列表。
- **`metadata_condition`** (object, 可选): 用于筛选知识块的元数据条件。
  - `conditions` (array): 元数据筛选条件列表。
    - `name` (string): 元数据字段名称 (例如, "author", "url")。
    - `comparison_operator` (string): 比较运算符 (例如, "=", "contains", ">")。
    - `value` (string): 要比较的值。

---

### 请求示例

```bash
curl --request POST \
     --url http://{address}/api/v1/retrieval \
     --header 'Content-Type: application/json' \
     --header 'Authorization: Bearer <YOUR_API_KEY>' \
     --data '
     {
          "question": "What is advantage of ragflow?",
          "dataset_ids": ["b2a62730759d11ef987d0242ac120004"],
          "document_ids": ["77df9ef4759a11ef8bdd0242ac120004"],
          "metadata_condition": {
            "conditions": [
              {
                "name": "author",
                "comparison_operator": "=",
                "value": "Toby"
              },
              {
                "name": "url",
                "comparison_operator": "not contains",
                "value": "amd"
              }
            ]
          }
     }'
```

---

### 响应

#### 成功

```json
{
    "code": 0,
    "data": {
        "chunks": [
            {
                "content": "ragflow content",
                "content_ltks": "ragflow content",
                "document_id": "5c5999ec7be811ef9cab0242ac120005",
                "document_keyword": "1.txt",
                "highlight": "<em>ragflow</em> content",
                "id": "d78435d142bd5cf6704da62c778795c5",
                "image_id": "",
                "important_keywords": [
                    ""
                ],
                "kb_id": "c7ee74067a2c11efb21c0242ac120006",
                "positions": [
                    ""
                ],
                "similarity": 0.9669436601210759,
                "term_similarity": 1.0,
                "vector_similarity": 0.8898122004035864
            }
        ],
        "doc_aggs": [
            {
                "count": 1,
                "doc_id": "5c5999ec7be811ef9cab0242ac120005",
                "doc_name": "1.txt"
            }
        ],
        "total": 1
    }
}
```

#### 失败

```json
{
    "code": 102,
    "message": "`datasets` is required."
}