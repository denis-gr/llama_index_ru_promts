"""Autoretriever prompts."""


from llama_index.prompts.base import PromptTemplate
from llama_index.prompts.prompt_type import PromptType
from llama_index.vector_stores.types import (
    ExactMatchFilter,
    MetadataInfo,
    VectorStoreInfo,
    VectorStoreQuerySpec,
)

# NOTE: these prompts are inspired from langchain's self-query prompt,
# and adapted to our use case.
# https://github.com/hwchase17/langchain/tree/main/langchain/chains/query_constructor/prompt.py


PREFIX = """\
Your goal is to structure the user's query to match the request schema provided below.

<< Structured Request Schema >>
When responding use a markdown code snippet with a JSON object formatted in the \
following schema:

{schema_str}

The query string should contain only text that is expected to match the contents of \
documents. Any conditions in the filter should not be mentioned in the query as well.

Make sure that filters only refer to attributes that exist in the data source.
Make sure that filters take into account the descriptions of attributes.
Make sure that filters are only used as needed. If there are no filters that should be \
applied return [] for the filter value.\

If the user's query explicitly mentions number of documents to retrieve, set top_k to \
that number, otherwise do not set top_k. 

"""


PREFIX = """\
Ваша цель - структурировать запрос пользователя в соответствии со схемой запроса, приведенной ниже.

<< Структурированная схема запроса >>
При ответе используйте фрагмент кода markdown с объектом JSON, отформатированным по
следующей схеме:

{schema_str}

Строка запроса должна содержать только текст, который, как ожидается, будет соответствовать содержимому \
documents. Какие-либо условия в фильтре также не должны упоминаться в запросе.

Убедитесь, что фильтры ссылаются только на атрибуты, существующие в источнике данных.
Убедитесь, что фильтры учитывают описания атрибутов.
Убедитесь, что фильтры используются только по мере необходимости. Если нет фильтров, которые должны быть \
применены, верните [] для значения фильтра.\

Если в запросе пользователя явно указано количество документов для извлечения, установите top_k равным \
этому числу, в противном случае не устанавливайте top_k. 

"""

example_info = VectorStoreInfo(
    content_info="Lyrics of a song",
    metadata_info=[
        MetadataInfo(name="artist", type="str", description="Name of the song artist"),
        MetadataInfo(
            name="genre",
            type="str",
            description='The song genre, one of "pop", "rock" or "rap"',
        ),
    ],
)

example_query = "What are songs by Taylor Swift or Katy Perry in the dance pop genre"

example_output = VectorStoreQuerySpec(
    query="teenager love",
    filters=[
        ExactMatchFilter(key="artist", value="Taylor Swift"),
        ExactMatchFilter(key="artist", value="Katy Perry"),
        ExactMatchFilter(key="genre", value="pop"),
    ],
)

EXAMPLES = (
    """\
<< Пример 1. >>
Источник данных:
``json
{info_str}
```

Пользовательский запрос:
{query_str}

Структурированный запрос:
``json
{output_str}
```
""".format(
        info_str=example_info.json(indent=4),
        query_str=example_query,
        output_str=example_output.json(),
    )
    .replace("{", "{{")
    .replace("}", "}}")
)


SUFFIX = """
<< Example 2. >>
Data Source:
```json
{info_str}
```

User Query:
{query_str}

Structured Request:
"""

СУФФИКС = """
<< Пример 2. >>
Источник данных:
``json
{info_str}
```

Пользовательский запрос:
{query_str}

Структурированный запрос:
"""

DEFAULT_VECTOR_STORE_QUERY_PROMPT_TMPL = PREFIX + EXAMPLES + SUFFIX


# deprecated, kept for backwards compatibility
"""Vector store query prompt."""
VectorStoreQueryPrompt = PromptTemplate

DEFAULT_VECTOR_STORE_QUERY_PROMPT = PromptTemplate(
    template=DEFAULT_VECTOR_STORE_QUERY_PROMPT_TMPL,
    prompt_type=PromptType.VECTOR_STORE_QUERY,
)
