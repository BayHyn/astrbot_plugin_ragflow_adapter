import json
from typing import TYPE_CHECKING, Any, Dict, List, Union
from astrbot.api import logger

if TYPE_CHECKING:
    from astrbot.api.provider import Provider


class QueryRewriterBase:
    """查询改写器基类"""

    def __init__(self, provider: "Provider"):
        if not provider:
            raise ValueError("Provider must be provided for QueryRewriter.")
        self.provider = provider

    async def _get_completion(self, prompt: str) -> str:
        """使用传入的 provider 生成回答"""
        logger.debug(f"发送到 LLM 的 Prompt: \n---PROMPT---\n{prompt}\n---END PROMPT---")
        try:
            resp = await self.provider.text_chat(prompt=prompt)
            completion_text = resp.completion_text.strip() if resp and resp.completion_text else ""
            logger.debug(f"从 LLM 收到的原始响应: {completion_text}")
            return completion_text
        except Exception as e:
            logger.error(f"调用 LLM Provider 时出错: {e}", exc_info=True)
            return ""

    async def rewrite(self, query: str, **kwargs) -> Any:
        """改写查询的抽象方法"""
        raise NotImplementedError("子类必须实现rewrite方法")


class ContextDependentRewriter(QueryRewriterBase):
    """上下文依赖型Query改写器"""

    async def rewrite(self, current_query: str, conversation_history: str) -> str:
        """上下文依赖型Query改写"""
        instruction = """
你是一个智能的查询优化助手。请分析用户的当前问题以及前序对话历史，判断当前问题是否依赖于上下文。
如果依赖，请将当前问题改写成一个独立的、包含所有必要上下文信息的完整问题。
如果不依赖，直接返回原问题。
"""
        prompt = f"""### 指令 ###
{instruction}

### 对话历史 ###
{conversation_history}

### 当前问题 ###
{current_query}

### 改写后的问题 ###
"""
        return await self._get_completion(prompt)


class ComparativeRewriter(QueryRewriterBase):
    """对比型Query改写器"""

    async def rewrite(self, query: str, context_info: str) -> str:
        """对比型Query改写"""
        instruction = """
你是一个查询分析专家。请分析用户的输入和相关的对话上下文，识别出问题中需要进行比较的多个对象。
然后，将原始问题改写成一个更明确、更适合在知识库中检索的对比性查询。
"""
        prompt = f"""### 指令 ###
{instruction}

### 对话历史/上下文信息 ###
{context_info}

### 原始问题 ###
{query}

### 改写后的查询 ###
"""
        return await self._get_completion(prompt)


class AmbiguousReferenceRewriter(QueryRewriterBase):
    """模糊指代型Query改写器"""

    async def rewrite(self, current_query: str, conversation_history: str) -> str:
        """模糊指代型Query改写"""
        instruction = """
你是一个消除语言歧义的专家。请分析用户的当前问题和对话历史，找出问题中 "都"、"它"、"这个" 等模糊指代词具体指向的对象。
然后，将这些指代词替换为明确的对象名称，生成一个清晰、无歧义的新问题。
"""
        prompt = f"""### 指令 ###
{instruction}

### 对话历史 ###
{conversation_history}

### 当前问题 ###
{current_query}

### 改写后的问题 ###
"""
        return await self._get_completion(prompt)


class MultiIntentRewriter(QueryRewriterBase):
    """多意图型Query改写器"""

    async def rewrite(self, query: str) -> List[str]:
        """多意图型Query改写 - 分解查询"""
        instruction = """
你是一个任务分解机器人。请将用户的复杂问题分解成多个独立的、可以单独回答的简单问题。以JSON数组格式输出。
"""
        prompt = f"""### 指令 ###
{instruction}

### 原始问题 ###
{query}

### 分解后的问题列表 ###
请以JSON数组格式输出，例如：["问题1", "问题2", "问题3"]
"""
        response = await self._get_completion(prompt)
        try:
            # The response might be in a markdown code block, so we need to clean it.
            cleaned_response = response.strip().replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned_response)
        except (json.JSONDecodeError, TypeError):
            return [response]


class RhetoricalRewriter(QueryRewriterBase):
    """反问型Query改写器"""

    async def rewrite(self, current_query: str, conversation_history: str) -> str:
        """反问型Query改写"""
        instruction = """
你是一个沟通理解大师。请分析用户的反问或带有情绪的陈述，识别其背后真实的意图和问题。
然后，将这个反问改写成一个中立、客观、可以直接用于知识库检索的问题。
"""
        prompt = f"""### 指令 ###
{instruction}

### 对话历史 ###
{conversation_history}

### 当前问题 ###
{current_query}

### 改写后的问题 ###
"""
        return await self._get_completion(prompt)


class QueryTypeDetector(QueryRewriterBase):
    """Query类型检测器"""

    async def detect(self, query: str, conversation_history: str = "", context_info: str = "") -> Dict[str, Any]:
        """自动识别Query类型"""
        instruction = """
你是一个智能的查询分析专家。请分析用户的查询，识别其属于以下哪种类型：
1. 上下文依赖型 - 包含"还有"、"其他"等需要上下文理解的词汇
2. 对比型 - 包含"哪个"、"比较"、"更"、"哪个更好"、"哪个更"等比较词汇
3. 模糊指代型 - 包含"它"、"他们"、"都"、"这个"等指代词
4. 多意图型 - 包含多个独立问题，用"、"或"？"分隔
5. 反问型 - 包含"不会"、"难道"等反问语气
6. 普通型 - 不属于以上任何一种的常规问题。

说明：如果同时存在多意图型、模糊指代型，优先级为多意图型>模糊指代型。请返回最符合的一种类型。

请严格按照以下JSON格式返回结果，不要包含任何其他解释：
{
    "query_type": "识别出的查询类型",
    "confidence": 0.8
}
"""
        prompt = f"""### 指令 ###
{instruction}

### 对话历史 ###
{conversation_history}

### 上下文信息 ###
{context_info}

### 原始查询 ###
{query}

### 分析结果 (JSON) ###
"""
        response = await self._get_completion(prompt)
        try:
            cleaned_response = response.strip().replace("```json", "").replace("```", "").strip()
            result = json.loads(cleaned_response)
            logger.debug(f"查询类型检测结果: {result}")
            return result
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"无法解析查询类型检测器的JSON响应: {response}")
            return {
                "query_type": "普通型",
                "confidence": 0.5
            }


class QueryRewriteManager:
    """Query改写管理器，负责协调各种改写器"""

    def __init__(self, provider: "Provider"):
        self.provider = provider
        self.context_rewriter = ContextDependentRewriter(provider)
        self.comparative_rewriter = ComparativeRewriter(provider)
        self.ambiguous_rewriter = AmbiguousReferenceRewriter(provider)
        self.multi_intent_rewriter = MultiIntentRewriter(provider)
        self.rhetorical_rewriter = RhetoricalRewriter(provider)
        self.type_detector = QueryTypeDetector(provider)

    async def rewrite_query(self, query: str, conversation_history: str = "") -> Union[str, List[str]]:
        """自动识别Query类型并进行改写"""
        detection_result = await self.type_detector.detect(query, conversation_history)
        query_type = detection_result.get('query_type', '普通型')
        logger.info(f"原始查询: '{query}', 检测到的类型: '{query_type}'")

        rewritten_query: Union[str, List[str]]

        if '上下文依赖' in query_type:
            rewritten_query = await self.context_rewriter.rewrite(query, conversation_history)
        elif '对比' in query_type:
            rewritten_query = await self.comparative_rewriter.rewrite(query, conversation_history)
        elif '模糊指代' in query_type:
            rewritten_query = await self.ambiguous_rewriter.rewrite(query, conversation_history)
        elif '多意图' in query_type:
            rewritten_query = await self.multi_intent_rewriter.rewrite(query)
        elif '反问' in query_type:
            rewritten_query = await self.rhetorical_rewriter.rewrite(query, conversation_history)
        else:
            # 对于“普通型”或无法识别的类型，返回原始查询
            rewritten_query = query
        
        logger.info(f"最终改写结果: {rewritten_query}")
        return rewritten_query
