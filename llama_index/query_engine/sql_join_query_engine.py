"""SQL Join query engine."""

import logging
from typing import Callable, Dict, Optional, Union

from llama_index.bridge.langchain import print_text
from llama_index.callbacks.base import CallbackManager
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.indices.query.query_transform.base import BaseQueryTransform
from llama_index.indices.query.schema import QueryBundle
from llama_index.indices.service_context import ServiceContext
from llama_index.indices.struct_store.sql_query import (
    BaseSQLTableQueryEngine,
    NLSQLTableQueryEngine,
)
from llama_index.llm_predictor import LLMPredictor
from llama_index.llm_predictor.base import BaseLLMPredictor
from llama_index.prompts.base import BasePromptTemplate, PromptTemplate
from llama_index.response.schema import RESPONSE_TYPE, Response
from llama_index.selectors.llm_selectors import LLMSingleSelector
from llama_index.selectors.pydantic_selectors import PydanticSingleSelector
from llama_index.selectors.utils import get_selector_from_context
from llama_index.tools.query_engine import QueryEngineTool

logger = logging.getLogger(__name__)


DEFAULT_SQL_JOIN_SYNTHESIS_PROMPT_TMPL =  """
Первоначальный вопрос приведен ниже.
Этот вопрос был переведен в SQL-запрос. Как SQL-запрос, так и \
ответ приведены ниже.
Учитывая ответ SQL, вопрос также был преобразован в более
подробный запрос
и выполнен с помощью другого механизма запросов.
Преобразованный запрос и ответ обработчика запросов также приведены ниже.
Учитывая SQL-запрос, SQL-ответ, преобразованный запрос и ответ механизма запросов, \
пожалуйста, синтезируйте ответ на исходный вопрос.

Оригинальный вопрос: {query_str}
SQL-запрос: {sql_query_str}
SQL-ответ: {sql_response_str}
Преобразованный запрос: {query_engine_query_str}
Ответ механизма запроса: {query_engine_response_str}
Ответ:
""" # noqa
DEFAULT_SQL_JOIN_SYNTHESIS_PROMPT = PromptTemplate(
    DEFAULT_SQL_JOIN_SYNTHESIS_PROMPT_TMPL
)


DEFAULT_SQL_AUGMENT_TRANSFORM_PROMPT_TMPL = """
"Первоначальный вопрос приведен ниже.
Этот вопрос был переведен в SQL-запрос. Как SQL-запрос, так и \
response приведены ниже.
SQL-ответ либо отвечает на вопрос, либо должен предоставлять дополнительный контекст \,
который можно использовать, чтобы сделать вопрос более конкретным.
Ваша задача состоит в том, чтобы придумать более конкретный вопрос, на который необходимо ответить, чтобы
полностью ответить на исходный вопрос, или "Нет", если на исходный вопрос уже
был получен полный ответ из SQL-ответа. Не создавайте новый вопрос, который является \
не имеет отношения к исходному вопросу; в этом случае верните вместо него None.

Примеры:

Оригинальный вопрос: Пожалуйста, дайте более подробную информацию о демографии города с
наибольшим населением.
SQL-запрос: ВЫБЕРИТЕ город, население ИЗ списка городов В ПОРЯДКЕ убывания численности населения, ОГРАНИЧЕНИЕ 1
Ответ SQL: Городом с самым высоким населением является Нью-Йорк.
Новый вопрос: Не могли бы вы рассказать мне больше о демографии Нью-Йорка?

Оригинальный вопрос: Пожалуйста, сравните спортивную среду городов Северной Америки.
SQL-запрос: ВЫБЕРИТЕ city_name ИЗ городов, ГДЕ continent = 'Северная Америка', ОГРАНИЧЕНИЕ 3
Ответ SQL: Городами в Северной Америке являются Нью-Йорк, Сан-Франциско и Торонто.
Новый вопрос: Какими видами спорта занимаются в Нью-Йорке, Сан-Франциско и Торонто?

Оригинальный вопрос: Какой город с самым большим населением?
SQL-запрос: ВЫБЕРИТЕ город, население ИЗ списка городов В ПОРЯДКЕ убывания численности населения, ОГРАНИЧЕНИЕ 1
Ответ SQL: Городом с самым высоким населением является Нью-Йорк.
Новый вопрос: Нет

Оригинальный вопрос: Из каких стран входят в тройку лучших игроков ATP?
SQL-запрос: ВЫБЕРИТЕ страну ИЗ списка игроков, рейтинг КОТОРЫХ <= 3
SQL-ответ: В тройку лучших игроков ATP входят представители Сербии, России и Испании.
Новый вопрос: Нет

Оригинальный вопрос: {query_str}
SQL-запрос: {sql_query_str}
SQL-ответ: {sql_response_str}
Новый вопрос: "
"""  # noqa
DEFAULT_SQL_AUGMENT_TRANSFORM_PROMPT = PromptTemplate(
    DEFAULT_SQL_AUGMENT_TRANSFORM_PROMPT_TMPL
)


def _default_check_stop(query_bundle: QueryBundle) -> bool:
    """Default check stop function."""
    return query_bundle.query_str.lower() == "none"


def _format_sql_query(sql_query: str) -> str:
    """Format SQL query."""
    return sql_query.replace("\n", " ").replace("\t", " ")


class SQLAugmentQueryTransform(BaseQueryTransform):
    """SQL Augment Query Transform.

    This query transform will transform the query into a more specific query
    after augmenting with SQL results.

    Args:
        llm_predictor (LLMPredictor): LLM predictor to use for query transformation.
        sql_augment_transform_prompt (BasePromptTemplate): PromptTemplate to use
            for query transformation.
        check_stop_parser (Optional[Callable[[str], bool]]): Check stop function.

    """

    def __init__(
        self,
        llm_predictor: Optional[BaseLLMPredictor] = None,
        sql_augment_transform_prompt: Optional[BasePromptTemplate] = None,
        check_stop_parser: Optional[Callable[[QueryBundle], bool]] = None,
    ) -> None:
        """Initialize params."""
        self._llm_predictor = llm_predictor or LLMPredictor()

        self._sql_augment_transform_prompt = (
            sql_augment_transform_prompt or DEFAULT_SQL_AUGMENT_TRANSFORM_PROMPT
        )
        self._check_stop_parser = check_stop_parser or _default_check_stop

    def _run(self, query_bundle: QueryBundle, metadata: Dict) -> QueryBundle:
        """Run query transform."""
        query_str = query_bundle.query_str
        sql_query = metadata["sql_query"]
        sql_query_response = metadata["sql_query_response"]
        new_query_str = self._llm_predictor.predict(
            self._sql_augment_transform_prompt,
            query_str=query_str,
            sql_query_str=sql_query,
            sql_response_str=sql_query_response,
        )
        return QueryBundle(
            new_query_str, custom_embedding_strs=query_bundle.custom_embedding_strs
        )

    def check_stop(self, query_bundle: QueryBundle) -> bool:
        """Check if query indicates stop."""
        return self._check_stop_parser(query_bundle)


class SQLJoinQueryEngine(BaseQueryEngine):
    """SQL Join Query Engine.

    This query engine can "Join" a SQL database results
    with another query engine.
    It can decide it needs to query the SQL database or the other query engine.
    If it decides to query the SQL database, it will first query the SQL database,
    whether to augment information with retrieved results from the other query engine.

    Args:
        sql_query_tool (QueryEngineTool): Query engine tool for SQL database.
            other_query_tool (QueryEngineTool): Other query engine tool.
        selector (Optional[Union[LLMSingleSelector, PydanticSingleSelector]]):
            Selector to use.
        service_context (Optional[ServiceContext]): Service context to use.
        sql_join_synthesis_prompt (Optional[BasePromptTemplate]):
            PromptTemplate to use for SQL join synthesis.
        sql_augment_query_transform (Optional[SQLAugmentQueryTransform]): Query
            transform to use for SQL augmentation.
        use_sql_join_synthesis (bool): Whether to use SQL join synthesis.
        callback_manager (Optional[CallbackManager]): Callback manager to use.
        verbose (bool): Whether to print intermediate results.

    """

    def __init__(
        self,
        sql_query_tool: QueryEngineTool,
        other_query_tool: QueryEngineTool,
        selector: Optional[Union[LLMSingleSelector, PydanticSingleSelector]] = None,
        service_context: Optional[ServiceContext] = None,
        sql_join_synthesis_prompt: Optional[BasePromptTemplate] = None,
        sql_augment_query_transform: Optional[SQLAugmentQueryTransform] = None,
        use_sql_join_synthesis: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = True,
    ) -> None:
        """Initialize params."""
        super().__init__(callback_manager=callback_manager)
        # validate that the query engines are of the right type
        if not isinstance(
            sql_query_tool.query_engine,
            (BaseSQLTableQueryEngine, NLSQLTableQueryEngine),
        ):
            raise ValueError(
                "sql_query_tool.query_engine must be an instance of "
                "BaseSQLTableQueryEngine or NLSQLTableQueryEngine"
            )
        self._sql_query_tool = sql_query_tool
        self._other_query_tool = other_query_tool

        sql_query_engine = sql_query_tool.query_engine
        self._service_context = service_context or sql_query_engine.service_context

        self._selector = selector or get_selector_from_context(
            self._service_context, is_multi=False
        )
        assert isinstance(self._selector, (LLMSingleSelector, PydanticSingleSelector))

        self._sql_join_synthesis_prompt = (
            sql_join_synthesis_prompt or DEFAULT_SQL_JOIN_SYNTHESIS_PROMPT
        )
        self._sql_augment_query_transform = (
            sql_augment_query_transform
            or SQLAugmentQueryTransform(
                llm_predictor=self._service_context.llm_predictor
            )
        )
        self._use_sql_join_synthesis = use_sql_join_synthesis
        self._verbose = verbose

    def _query_sql_other(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Query SQL database + other query engine in sequence."""
        # first query SQL database
        sql_response = self._sql_query_tool.query_engine.query(query_bundle)
        if not self._use_sql_join_synthesis:
            return sql_response

        sql_query = (
            sql_response.metadata["sql_query"] if sql_response.metadata else None
        )
        if self._verbose:
            print_text(f"SQL query: {sql_query}\n", color="yellow")
            print_text(f"SQL response: {sql_response}\n", color="yellow")

        # given SQL db, transform query into new query
        new_query = self._sql_augment_query_transform(
            query_bundle.query_str,
            metadata={
                "sql_query": _format_sql_query(sql_query),
                "sql_query_response": str(sql_response),
            },
        )

        if self._verbose:
            print_text(
                f"Transformed query given SQL response: {new_query.query_str}\n",
                color="blue",
            )
        logger.info(f"> Transformed query given SQL response: {new_query.query_str}")
        if self._sql_augment_query_transform.check_stop(new_query):
            return sql_response

        other_response = self._other_query_tool.query_engine.query(new_query)
        if self._verbose:
            print_text(f"query engine response: {other_response}\n", color="pink")
        logger.info(f"> query engine response: {other_response}")

        response_str = self._service_context.llm_predictor.predict(
            self._sql_join_synthesis_prompt,
            query_str=query_bundle.query_str,
            sql_query_str=sql_query,
            sql_response_str=str(sql_response),
            query_engine_query_str=new_query.query_str,
            query_engine_response_str=str(other_response),
        )
        if self._verbose:
            print_text(f"Final response: {response_str}\n", color="green")
        response_metadata = {
            **(sql_response.metadata or {}),
            **(other_response.metadata or {}),
        }
        source_nodes = other_response.source_nodes
        return Response(
            response_str,
            metadata=response_metadata,
            source_nodes=source_nodes,
        )

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Query and get response."""
        # TODO: see if this can be consolidated with logic in RouterQueryEngine
        metadatas = [self._sql_query_tool.metadata, self._other_query_tool.metadata]
        result = self._selector.select(metadatas, query_bundle)
        # pick sql query
        if result.ind == 0:
            if self._verbose:
                print_text(f"Querying SQL database: {result.reason}\n", color="blue")
            logger.info(f"> Querying SQL database: {result.reason}")
            return self._query_sql_other(query_bundle)
        elif result.ind == 1:
            if self._verbose:
                print_text(
                    f"Querying other query engine: {result.reason}\n", color="blue"
                )
            logger.info(f"> Querying other query engine: {result.reason}")
            response = self._other_query_tool.query_engine.query(query_bundle)
            if self._verbose:
                print_text(f"Query Engine response: {response}\n", color="pink")
            return response
        else:
            raise ValueError(f"Invalid result.ind: {result.ind}")

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        # TODO: make async
        return self._query(query_bundle)
