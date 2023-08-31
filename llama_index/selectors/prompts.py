from llama_index.prompts.base import PromptTemplate
from llama_index.prompts.prompt_type import PromptType

"""Single select prompt.

PromptTemplate to select one out of `num_choices` options provided in `context_list`,
given a query `query_str`.

Required template variables: `num_chunks`, `context_list`, `query_str`

"""
SingleSelectPrompt = PromptTemplate

"""Multiple select prompt.

PromptTemplate to select multiple candidates (up to `max_outputs`) out of `num_choices`
options provided in `context_list`, given a query `query_str`.

Required template variables: `num_chunks`, `context_list`, `query_str`,
    `max_outputs`
"""
MultiSelectPrompt = PromptTemplate


# single select
DEFAULT_SINGLE_SELECT_PROMPT_TMPL = (

    "Ниже приведены некоторые варианты. Он представлен в виде пронумерованного списка "
    "(от 1 до {num_choices}),"
    "где каждому элементу в списке соответствует краткое описание.\n"
    "---------------------\ n"
    "{context_list}"
    "\n---------------------\ n"
    "Используя только указанные выше варианты, а не предварительные знания, верните "
    "выбор, который наиболее релевантен для вопроса: '{query_str}'\n"
)


DEFAULT_SINGLE_SELECT_PROMPT = PromptTemplate(
    template=DEFAULT_SINGLE_SELECT_PROMPT_TMPL, prompt_type=PromptType.SINGLE_SELECT
)


# multiple select
DEFAULT_MULTI_SELECT_PROMPT_TMPL = (
"Ниже приведены некоторые варианты. Он представлен в пронумерованном "
    "список (от 1 до {num_choices})",
    "где каждому элементу в списке соответствует краткое описание.\n"
    "---------------------\ n"
    "{context_list}"
    "\n---------------------\ n"
    "Используя только приведенные выше варианты, а не предварительные знания, верните лучшие варианты "
    "(не более {max_outputs}, но выберите только то, что необходимо), что "
    "наиболее релевантны для вопроса: '{query_str}'\n"
)



DEFAULT_MULTIPLE_SELECT_PROMPT = PromptTemplate(
    template=DEFAULT_MULTI_SELECT_PROMPT_TMPL, prompt_type=PromptType.MULTI_SELECT
)

# single pydantic select
DEFAULT_SINGLE_PYD_SELECT_PROMPT_TMPL = (
"Ниже приведены некоторые варианты. Он представлен в виде пронумерованного списка "
"(от 1 до {num_choices}),"
"где каждому элементу в списке соответствует краткое описание.\n"
"---------------------\ n"
"{context_list}"
"\n---------------------\ n"
"Используя только описанные выше варианты, а не предварительные знания, сгенерируйте "
"объект выбора и причина, которые наиболее релевантны для "
"вопрос: '{query_str}'\n"
)


# multiple pydantic select
DEFAULT_MULTI_PYD_SELECT_PROMPT_TMPL = (
"Ниже приведены некоторые варианты. Он представлен в пронумерованном "
    "список (от 1 до {num_choices})",
    "где каждому элементу в списке соответствует краткое описание.\n"
    "---------------------\ n"
    "{context_list}"
    "\n---------------------\ n"
    "Используя только приведенные выше варианты, а не предварительные знания, верните лучший вариант (ы) "
    "(не более {max_outputs}, но выберите только то, что необходимо) путем генерации "
    "объект отбора и причины, которые наиболее релевантны для "
    "вопрос: '{query_str}'\n"
)
