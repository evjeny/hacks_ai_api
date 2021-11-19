# hacks_ai_api
API for Hacks-AI

### `/classify/{text}`

Параметры:
* `text` -- описание товара, для которого нужно предсказать категорию

Что возвращает:
* json с полями:
  - `category`: код категории товара, строка
  - `category_description`: описание категории товара, строка
  - `probability`: уверенность модели в своем предсказании, дробное число от 0 до 1
  - `parent_category`: код категории родителя товара
  - `parent_category_description`: описание категории родительского товара, если категория не найдена, то возвращает "Нет описания"
