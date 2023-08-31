""" Knowledge Graph Query Engine"""

import logging
from typing import Any, List, Optional, Sequence

from llama_index.bridge.langchain import print_text
from llama_index.callbacks.schema import CBEventType, EventPayload
from llama_index.graph_stores.registery import (
    GRAPH_STORE_CLASS_TO_GRAPH_STORE_TYPE,
    GraphStoreType,
)
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.indices.query.schema import QueryBundle
from llama_index.indices.service_context import ServiceContext
from llama_index.prompts.base import BasePromptTemplate, PromptTemplate, PromptType
from llama_index.response.schema import RESPONSE_TYPE
from llama_index.response_synthesizers import BaseSynthesizer, get_response_synthesizer
from llama_index.schema import NodeWithScore, TextNode
from llama_index.storage.storage_context import StorageContext

logger = logging.getLogger(__name__)

# Prompt
DEFAULT_NEBULAGRAPH_NL2CYPHER_PROMPT_TMPL = """
Сгенерируйте запрос Nebula Graph на естественном языке.
Используйте только предоставленные типы связей и свойства в схеме.
Не используйте какие-либо другие типы связей или свойства, которые не предусмотрены.
Схема:
---
{schema}
---
Примечание: Небулаграф говорит на диалекте сайфера, по сравнению со стандартным сайфером:

1. для сравнения используется двойной знак равенства: `==` вместо `=`
2. требуется явное указание метки при обращении к свойствам узла, т.е.
v - это переменная узла, и мы знаем, что ее метка - Foo, v.'foo`.имя правильное
, в то время как v.name это не так.

Например, смотрите это различие между стандартным диалектом шифрования и графом туманности:
`разница
< СОВПАДЕНИЕ (p:персона) -[:режиссер]->(m:фильм), ГДЕ m.name = "Крестный отец"
< ВОЗВРАТ p.name;
---
> СОВПАДЕНИЕ (p:`персона`) -[:режиссер]->(m:`фильм`) ГДЕ m.`фильм`.`название` == 'Крестный отец'
> ВЕРНУТЬ p.`лицо`.`имя`;
```

Вопрос: {query_str}

Запрос на диалект шифра графа туманности:
"""
DEFAULT_NEBULAGRAPH_NL2CYPHER_PROMPT = PromptTemplate(
    DEFAULT_NEBULAGRAPH_NL2CYPHER_PROMPT_TMPL,
    prompt_type=PromptType.TEXT_TO_GRAPH_QUERY,
)

# Prompt
DEFAULT_NEO4J_NL2CYPHER_PROMPT_TMPL = (
    "Задача: Сгенерировать инструкцию Cypher для запроса базы данных graph.\n"
    "Инструкции:\n"
    "Используйте только предоставленные типы связей и свойства в схеме.\n"
    "Не используйте какие-либо другие типы отношений или свойства, которые не предусмотрены.\n"
    "Схема:\n"
    "{schema}\n"
    "Примечание: Не включайте никаких объяснений или извинений в свои ответы.\n"
    "Не отвечайте ни на какие вопросы, которые могут касаться чего-либо другого, кроме вас"
    "чтобы сконструировать шифровальное выражение. \n"
    "Не включайте никакого текста, кроме сгенерированного оператора шифрования.\n"
    "\n"
    "Вопрос в следующем:\n"
    "{query_str}\n"
)

DEFAULT_NEO4J_NL2CYPHER_PROMPT = PromptTemplate(
    DEFAULT_NEO4J_NL2CYPHER_PROMPT_TMPL,
    prompt_type=PromptType.TEXT_TO_GRAPH_QUERY,
)

DEFAULT_NL2GRAPH_PROMPT_MAP = {
    GraphStoreType.NEBULA: DEFAULT_NEBULAGRAPH_NL2CYPHER_PROMPT,
    GraphStoreType.NEO4J: DEFAULT_NEO4J_NL2CYPHER_PROMPT,
}

DEFAULT_KG_RESPONSE_ANSWER_PROMPT_TMPL = """Первоначальный вопрос приведен ниже.
Этот вопрос был преобразован в запрос к базе данных Graph.
Как графический запрос, так и ответ приведены ниже.
Учитывая ответ на запрос Graph, синтезируйте ответ на исходный вопрос.

Оригинальный вопрос: {query_str}
Графический запрос: {kg_query_str}
Графический ответ: {kg_response_str}
Ответ:
"""

DEFAULT_KG_RESPONSE_ANSWER_PROMPT = PromptTemplate(
    DEFAULT_KG_RESPONSE_ANSWER_PROMPT_TMPL,
    prompt_type=PromptType.QUESTION_ANSWER,
)


class KnowledgeGraphQueryEngine(BaseQueryEngine):
    """Knowledge graph query engine.

    Query engine to call a knowledge graph.

    Args:
        service_context (Optional[ServiceContext]): A service context to use.
        storage_context (Optional[StorageContext]): A storage context to use.
        refresh_schema (bool): Whether to refresh the schema.
        verbose (bool): Whether to print intermediate results.
        response_synthesizer (Optional[BaseSynthesizer]):
            A BaseSynthesizer object.
        **kwargs: Additional keyword arguments.

    """

    def __init__(
        self,
        service_context: Optional[ServiceContext] = None,
        storage_context: Optional[StorageContext] = None,
        graph_query_synthesis_prompt: Optional[BasePromptTemplate] = None,
        graph_response_answer_prompt: Optional[BasePromptTemplate] = None,
        refresh_schema: bool = False,
        verbose: bool = False,
        response_synthesizer: Optional[BaseSynthesizer] = None,
        **kwargs: Any,
    ):
        # Ensure that we have a graph store
        assert storage_context is not None, "Must provide a storage context."
        assert (
            storage_context.graph_store is not None
        ), "Must provide a graph store in the storage context."
        self._storage_context = storage_context
        self.graph_store = storage_context.graph_store

        self._service_context = service_context or ServiceContext.from_defaults()

        # Get Graph Store Type
        self._graph_store_type = GRAPH_STORE_CLASS_TO_GRAPH_STORE_TYPE[
            self.graph_store.__class__
        ]

        # Get Graph schema
        self._graph_schema = self.graph_store.get_schema(refresh=refresh_schema)

        # Get graph store query synthesis prompt
        self._graph_query_synthesis_prompt = (
            graph_query_synthesis_prompt
            or DEFAULT_NL2GRAPH_PROMPT_MAP[self._graph_store_type]
        )

        self._graph_response_answer_prompt = (
            graph_response_answer_prompt or DEFAULT_KG_RESPONSE_ANSWER_PROMPT
        )
        self._verbose = verbose
        self._response_synthesizer = response_synthesizer or get_response_synthesizer(
            callback_manager=self._service_context.callback_manager,
            service_context=self._service_context,
        )

        super().__init__(self._service_context.callback_manager)

    def generate_query(self, query_str: str) -> str:
        """Generate a Graph Store Query from a query bundle."""
        # Get the query engine query string

        graph_store_query: str = self._service_context.llm_predictor.predict(
            self._graph_query_synthesis_prompt,
            query_str=query_str,
            schema=self._graph_schema,
        )

        return graph_store_query

    async def agenerate_query(self, query_str: str) -> str:
        """Generate a Graph Store Query from a query bundle."""
        # Get the query engine query string

        graph_store_query: str = await self._service_context.llm_predictor.apredict(
            self._graph_query_synthesis_prompt,
            query_str=query_str,
            schema=self._graph_schema,
        )

        return graph_store_query

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Get nodes for response."""
        graph_store_query = self.generate_query(query_bundle.query_str)
        if self._verbose:
            print_text(f"Graph Store Query:\n{graph_store_query}\n", color="yellow")
        logger.debug(f"Graph Store Query:\n{graph_store_query}")

        with self.callback_manager.event(
            CBEventType.RETRIEVE,
            payload={EventPayload.QUERY_STR: graph_store_query},
        ) as retrieve_event:
            # Get the graph store response
            graph_store_response = self.graph_store.query(query=graph_store_query)
            if self._verbose:
                print_text(
                    f"Graph Store Response:\n{graph_store_response}\n",
                    color="yellow",
                )
            logger.debug(f"Graph Store Response:\n{graph_store_response}")

            retrieve_event.on_end(payload={EventPayload.RESPONSE: graph_store_response})

        retrieved_graph_context: Sequence = self._graph_response_answer_prompt.format(
            query_str=query_bundle.query_str,
            kg_query_str=graph_store_query,
            kg_response_str=graph_store_response,
        )

        node = NodeWithScore(
            node=TextNode(
                text=retrieved_graph_context,
                score=1.0,
                metadata={
                    "query_str": query_bundle.query_str,
                    "graph_store_query": graph_store_query,
                    "graph_store_response": graph_store_response,
                    "graph_schema": self._graph_schema,
                },
            )
        )
        return [node]

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Query the graph store."""
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            nodes: List[NodeWithScore] = self._retrieve(query_bundle)

            response = self._response_synthesizer.synthesize(
                query=query_bundle,
                nodes=nodes,
            )

            if self._verbose:
                print_text(f"Final Response: {response}\n", color="green")

            query_event.on_end(payload={EventPayload.RESPONSE: response})

        return response

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        graph_store_query = await self.agenerate_query(query_bundle.query_str)
        if self._verbose:
            print_text(f"Graph Store Query:\n{graph_store_query}\n", color="yellow")
        logger.debug(f"Graph Store Query:\n{graph_store_query}")

        with self.callback_manager.event(
            CBEventType.RETRIEVE,
            payload={EventPayload.QUERY_STR: graph_store_query},
        ) as retrieve_event:
            # Get the graph store response
            # TBD: This is a blocking call. We need to make it async.
            graph_store_response = self.graph_store.query(query=graph_store_query)
            if self._verbose:
                print_text(
                    f"Graph Store Response:\n{graph_store_response}\n",
                    color="yellow",
                )
            logger.debug(f"Graph Store Response:\n{graph_store_response}")

            retrieve_event.on_end(payload={EventPayload.RESPONSE: graph_store_response})

        retrieved_graph_context: Sequence = self._graph_response_answer_prompt.format(
            query_str=query_bundle.query_str,
            kg_query_str=graph_store_query,
            kg_response_str=graph_store_response,
        )

        node = NodeWithScore(
            node=TextNode(
                text=retrieved_graph_context,
                score=1.0,
                metadata={
                    "query_str": query_bundle.query_str,
                    "graph_store_query": graph_store_query,
                    "graph_store_response": graph_store_response,
                    "graph_schema": self._graph_schema,
                },
            )
        )
        return [node]

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Query the graph store."""
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            nodes = await self._aretrieve(query_bundle)
            response = await self._response_synthesizer.asynthesize(
                query=query_bundle,
                nodes=nodes,
            )

            if self._verbose:
                print_text(f"Final Response: {response}\n", color="green")

            query_event.on_end(payload={EventPayload.RESPONSE: response})

        return response
