"""SQL Vector query engine."""

import logging
from typing import Any, Optional, Union

from llama_index.callbacks.base import CallbackManager
from llama_index.indices.service_context import ServiceContext
from llama_index.indices.struct_store.sql_query import (
    BaseSQLTableQueryEngine,
    NLSQLTableQueryEngine,
)
from llama_index.indices.vector_store.retrievers.auto_retriever import (
    VectorIndexAutoRetriever,
)
from llama_index.prompts.base import BasePromptTemplate, PromptTemplate
from llama_index.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.query_engine.sql_join_query_engine import (
    SQLAugmentQueryTransform,
    SQLJoinQueryEngine,
)
from llama_index.selectors.llm_selectors import LLMSingleSelector
from llama_index.selectors.pydantic_selectors import PydanticSingleSelector
from llama_index.tools.query_engine import QueryEngineTool

logger = logging.getLogger(__name__)


DEFAULT_SQL_VECTOR_SYNTHESIS_PROMPT_TMPL = """
Первоначальный вопрос приведен ниже.
Этот вопрос был переведен в SQL-запрос. \
Как SQL-запрос, так и ответ приведены ниже.
Учитывая ответ SQL, вопрос также был переведен в запрос векторного хранилища.
Запрос и ответ векторного хранилища приведены ниже.
Учитывая SQL-запрос, SQL-ответ, преобразованный запрос векторного хранилища и векторное хранилище \
ответ, пожалуйста, синтезируйте ответ на исходный вопрос.

Оригинальный вопрос: {query_str}
SQL-запрос: {sql_query_str}
SQL-ответ: {sql_response_str}
Преобразованный запрос векторного хранилища: {query_engine_query_str}
Ответ векторного хранилища: {query_engine_response_str}
Ответ:
"""  # noqa
DEFAULT_SQL_VECTOR_SYNTHESIS_PROMPT = PromptTemplate(
    DEFAULT_SQL_VECTOR_SYNTHESIS_PROMPT_TMPL
)


# NOTE: maintain for backwards compatibility
class SQLAutoVectorQueryEngine(SQLJoinQueryEngine):
    """SQL + Vector Index Auto Retriever Query Engine.

    This query engine can query both a SQL database
    as well as a vector database. It will first decide
    whether it needs to query the SQL database or vector store.
    If it decides to query the SQL database, it will also decide
    whether to augment information with retrieved results from the vector store.
    We use the VectorIndexAutoRetriever to retrieve results.

    Args:
        sql_query_tool (QueryEngineTool): Query engine tool for SQL database.
        vector_query_tool (QueryEngineTool): Query engine tool for vector database.
        selector (Optional[Union[LLMSingleSelector, PydanticSingleSelector]]):
            Selector to use.
        service_context (Optional[ServiceContext]): Service context to use.
        sql_vector_synthesis_prompt (Optional[BasePromptTemplate]):
            Prompt to use for SQL vector synthesis.
        sql_augment_query_transform (Optional[SQLAugmentQueryTransform]): Query
            transform to use for SQL augmentation.
        use_sql_vector_synthesis (bool): Whether to use SQL vector synthesis.
        callback_manager (Optional[CallbackManager]): Callback manager to use.
        verbose (bool): Whether to print intermediate results.

    """

    def __init__(
        self,
        sql_query_tool: QueryEngineTool,
        vector_query_tool: QueryEngineTool,
        selector: Optional[Union[LLMSingleSelector, PydanticSingleSelector]] = None,
        service_context: Optional[ServiceContext] = None,
        sql_vector_synthesis_prompt: Optional[BasePromptTemplate] = None,
        sql_augment_query_transform: Optional[SQLAugmentQueryTransform] = None,
        use_sql_vector_synthesis: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = True,
    ) -> None:
        """Initialize params."""
        # validate that the query engines are of the right type
        if not isinstance(
            sql_query_tool.query_engine,
            (BaseSQLTableQueryEngine, NLSQLTableQueryEngine),
        ):
            raise ValueError(
                "sql_query_tool.query_engine must be an instance of "
                "BaseSQLTableQueryEngine or NLSQLTableQueryEngine"
            )
        if not isinstance(vector_query_tool.query_engine, RetrieverQueryEngine):
            raise ValueError(
                "vector_query_tool.query_engine must be an instance of "
                "RetrieverQueryEngine"
            )
        if not isinstance(
            vector_query_tool.query_engine.retriever, VectorIndexAutoRetriever
        ):
            raise ValueError(
                "vector_query_tool.query_engine.retriever must be an instance "
                "of VectorIndexAutoRetriever"
            )

        sql_vector_synthesis_prompt = (
            sql_vector_synthesis_prompt or DEFAULT_SQL_VECTOR_SYNTHESIS_PROMPT
        )
        super().__init__(
            sql_query_tool,
            vector_query_tool,
            selector=selector,
            service_context=service_context,
            sql_join_synthesis_prompt=sql_vector_synthesis_prompt,
            sql_augment_query_transform=sql_augment_query_transform,
            use_sql_join_synthesis=use_sql_vector_synthesis,
            callback_manager=callback_manager,
            verbose=verbose,
        )

    @classmethod
    def from_sql_and_vector_query_engines(
        cls,
        sql_query_engine: Union[BaseSQLTableQueryEngine, NLSQLTableQueryEngine],
        sql_tool_name: str,
        sql_tool_description: str,
        vector_auto_retriever: RetrieverQueryEngine,
        vector_tool_name: str,
        vector_tool_description: str,
        selector: Optional[Union[LLMSingleSelector, PydanticSingleSelector]] = None,
        **kwargs: Any,
    ) -> "SQLAutoVectorQueryEngine":
        """From SQL and vector query engines.

        Args:
            sql_query_engine (BaseSQLTableQueryEngine): SQL query engine.
            vector_query_engine (VectorIndexAutoRetriever): Vector retriever.
            selector (Optional[Union[LLMSingleSelector, PydanticSingleSelector]]):
                Selector to use.

        """
        sql_query_tool = QueryEngineTool.from_defaults(
            sql_query_engine, name=sql_tool_name, description=sql_tool_description
        )
        vector_query_tool = QueryEngineTool.from_defaults(
            vector_auto_retriever,
            name=vector_tool_name,
            description=vector_tool_description,
        )
        return cls(sql_query_tool, vector_query_tool, selector, **kwargs)
