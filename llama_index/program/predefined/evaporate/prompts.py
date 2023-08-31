"""Prompts from evaporate repo.


Full credits go to: https://github.com/HazyResearch/evaporate


"""

from llama_index.prompts import PromptTemplate

# deprecated, kept for backward compatibility

"""Pandas PromptTemplate. Convert query to python code.

Required template variables: `chunk`, `topic`.

Args:
    template (str): Template for the PromptTemplate.
    **prompt_kwargs: Keyword arguments for the PromptTemplate.

"""
SchemaIDPrompt = PromptTemplate

"""Function generation PromptTemplate. Generate a function from existing text.

Required template variables: `context_str`, `query_str`,
    `attribute`, `function_field`.

Args:
    template (str): Template for the PromptTemplate.
    **prompt_kwargs: Keyword arguments for the PromptTemplate.

"""
FnGeneratePrompt = PromptTemplate

# used for schema identification
SCHEMA_ID_PROMPT_TMPL = f"""Sample text:
<tr class="mergedrow"><th scope="row" class="infobox-label"><div style="text-indent:-0.9em;margin-left:1.2em;font-weight:normal;">•&nbsp;<a href="/wiki/Monarchy_of_Canada" title="Monarchy of Canada">Monarch</a> </div></th><td class="infobox-data"><a href="/wiki/Charles_III" title="Charles III">Charles III</a></td></tr>
<tr class="mergedrow"><th scope="row" class="infobox-label"><div style="text-indent:-0.9em;margin-left:1.2em;font-weight:normal;">•&nbsp;<span class="nowrap"><a href="/wiki/Governor_General_of_Canada" title="Governor General of Canada">Governor General</a></span> </div></th><td class="infobox-data"><a href="/wiki/Mary_Simon" title="Mary Simon">Mary Simon</a></td></tr>
<b>Provinces and Territories</b class='navlinking countries'>
<ul>
<li>Saskatchewan</li>
<li>Manitoba</li>
<li>Ontario</li>
<li>Quebec</li>
<li>New Brunswick</li>
<li>Prince Edward Island</li>
<li>Nova Scotia</li>
<li>Newfoundland and Labrador</li>
<li>Yukon</li>
<li>Nunavut</li>
<li>Northwest Territories</li>
</ul>

Question: List all relevant attributes about 'Canada' that are exactly mentioned in this sample text if any.
Answer: 
- Monarch: Charles III
- Governor General: Mary Simon
- Provinces and Territories: Saskatchewan, Manitoba, Ontario, Quebec, New Brunswick, Prince Edward Island, Nova Scotia, Newfoundland and Labrador, Yukon, Nunavut, Northwest Territories

----

Sample text:
Patient birth date: 1990-01-01
Prescribed medication: aspirin, ibuprofen, acetaminophen
Prescribed dosage: 1 tablet, 2 tablets, 3 tablets
Doctor's name: Dr. Burns
Date of discharge: 2020-01-01
Hospital address: 123 Main Street, New York, NY 10001

Question: List all relevant attributes about 'medications' that are exactly mentioned in this sample text if any.
Answer: 
- Prescribed medication: aspirin, ibuprofen, acetaminophen
- Prescribed dosage: 1 tablet, 2 tablets, 3 tablets

----

Sample text:
{{chunk:}}

Question: List all relevant attributes about '{{topic:}}' that are exactly mentioned in this sample text if any. 
Answer:"""  # noqa: E501, F541

SCHEMA_ID_PROMPT_TMPL = f"""Пример текста:
<tr class="mergedrow"><th scope="row" class="infobox-label"><div style="отступ текста:-0,9em;поле слева:1,2em;вес шрифта:обычный;">•&nbsp;<a href="/wiki/Monarchy_of_Canada" title="Монархия Канады">Монарх</a> </div></th><td class="infobox-data"><a href="/wiki/Charles_III" title="Карл III">Чарльз III</a></td></tr>
<tr class="mergedrow"><th scope="row" class="infobox-label"><div style="отступ текста:-0,9em;поле слева:1,2em;вес шрифта:обычный;">•&nbsp;<span class="nowrap"><a href="/wiki/Governor_General_of_Canada" title="Генерал-губернатор Канады">Генерал-губернатор</a></span> </div></th><td class="infobox-data"><a href="/wiki/Mary_Simon" title="Мэри Саймон">Мэри Саймон</a></td></tr>
<b>Провинции и территории</b class='соединяющие страны'>
<ул>
<li>Саскачеван</li>
<li>Манитоба</li>
<li>Онтарио</li>
<li>Квебек</li>
<li>Нью-Брансуик</li>
<li>Остров Принца Эдуарда</li>
<li>Новая Шотландия</li>
<li>Ньюфаундленд и Лабрадор</li>
<li>Юкон</li>
<li>Нунавут</li>
<li>Северо-западные территории</li>
</ul>

Вопрос: Перечислите все соответствующие атрибуты "Канады", которые точно упоминаются в этом образце текста, если таковые имеются.
Ответ: 
- Монарх: Карл III
- Генерал-губернатор: Мэри Саймон
- Провинции и территории: Саскачеван, Манитоба, Онтарио, Квебек, Нью-Брансуик, остров Принца Эдуарда, Новая Шотландия, Ньюфаундленд и Лабрадор, Юкон, Нунавут, Северо-Западные территории

----

Пример текста:
Дата рождения пациента: 1990-01-01
Предписанные лекарства: аспирин, ибупрофен, ацетаминофен
Предписанная дозировка: 1 таблетка, 2 таблетки, 3 таблетки
Имя врача: доктор Бернс
Дата выписки: 2020-01-01
Адрес больницы: 123 Main Street, Нью-Йорк, NY 10001

Вопрос: Перечислите все соответствующие атрибуты "лекарств", которые точно упоминаются в этом образце текста, если таковые имеются.
Ответ: 
- Назначенные лекарства: аспирин, ибупрофен, ацетаминофен
- Предписанная дозировка: 1 таблетка, 2 таблетки, 3 таблетки

----

Пример текста:
{{chunk:}}

Вопрос: Перечислите все соответствующие атрибуты '{{topic:}}', которые точно упоминаются в этом примере текста, если таковые имеются. 
Ответ: """

SCHEMA_ID_PROMPT = PromptTemplate(SCHEMA_ID_PROMPT_TMPL)


# used for function generation

FN_GENERATION_PROMPT_TMPL = f"""Here is a sample of text:

{{context_str:}}


Question: {{query_str:}}

Given the function signature, write Python code to extract the 
"{{attribute:}}" field from the text.
Return the result as a single value (string, int, float), and not a list.
Make sure there is a return statement in the code. Do not leave out a return statement.
{{expected_output_str:}}

import re

def get_{{function_field:}}_field(text: str):
    \"""
    Function to extract the "{{attribute:}} field", and return the result 
    as a single value.
    \"""
    """  # noqa: E501, F541

FN_GENERATION_PROMPT_TMPL = f"""Вот пример текста:

{{context_str:}}


Вопрос: {{query_str:}}

Учитывая сигнатуру функции, напишите код на Python, чтобы извлечь поле
"{{attribute:}}" из текста.
Возвращает результат в виде одного значения (string, int, float), а не списка.
Убедитесь, что в коде есть оператор return. Не оставляйте без внимания заявление о возврате.
{{expected_output_str:}}

импортировать повторно

def get_{{function_field:}}_field(text: str):
    \"""
    Функция для извлечения "{{attribute:}} поле" и возврата результата
в виде одного значения.
    \"""
    """

FN_GENERATION_PROMPT = PromptTemplate(FN_GENERATION_PROMPT_TMPL)


FN_GENERATION_LIST_PROMPT_TMPL = f"""Here is a sample of text:

{{context_str:}}


Question: {{query_str:}}

Given the function signature, write Python code to extract the 
"{{attribute:}}" field from the text.
Return the result as a list of values (if there is just one item, return a single \
element list).
Make sure there is a return statement in the code. Do not leave out a return statement.
{{expected_output_str:}}

import re

def get_{{function_field:}}_field(text: str) -> List:
    \"""
    Function to extract the "{{attribute:}} field", and return the result 
    as a single value.
    \"""
    """  # noqa: E501, F541

f"""Вот образец текста:

{{context_str:}}


Вопрос: {{query_str:}}

Учитывая сигнатуру функции, напишите код на Python, чтобы извлечь поле
"{{attribute:}}" из текста.
Верните результат в виде списка значений (если есть только один элемент, верните один \
список элементов).
Убедитесь, что в коде есть оператор return. Не оставляйте без внимания заявление о возврате.
{{expected_output_str:}}

импортировать повторно

def get_{{function_field:}}_field(text: str) -> Список:
    \"""
    Функция для извлечения "{{attribute:}} поле" и возврата результата
в виде одного значения.
    \"""
    """

FN_GENERATION_LIST_PROMPT = PromptTemplate(FN_GENERATION_LIST_PROMPT_TMPL)

DEFAULT_EXPECTED_OUTPUT_PREFIX_TMPL = (
    "Here is the expected output on the text after running the function. "
    "Please do not write a function that would return a different output. "
    "Expected output: "
)


DEFAULT_FIELD_EXTRACT_QUERY_TMPL = (
    'Write a python function to extract the entire "{field}" field from text, '
    "but not any other metadata. Return the result as a list."
)

DEFAULT_EXPECTED_OUTPUT_PREFIX_TMPL = (
"Вот ожидаемый вывод текста после запуска функции."
    "Пожалуйста, не пишите функцию, которая возвращала бы другой результат."
    "Ожидаемый результат: "
)


DEFAULT_FIELD_EXTRACT_QUERY_TMPL = (
    'Напишите функцию python для извлечения всего поля "{field}" из текста, '
    "но не какие-либо другие метаданные. Верните результат в виде списка."
)