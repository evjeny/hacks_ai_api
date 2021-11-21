# hacks_ai_api
API for Hacks-AI

Веса для моделек: [Google Drive](https://drive.google.com/drive/folders/1wbyiRsnuBOEGTOXCsZDekEJCJ8gLSMFN)

Установка зависимостей: Anaconda 4.10.1: `conda env create -f environment.yml`

Попробовать функции API можно по ссылке: `http://localhost:8000/docs`

### `/classify/{text}`

Параметры:
* `text` -- описание товара, для которого нужно предсказать категорию

Что возвращает:
* json с полем `categories`, список из возможных категорий, каждая категория имеет поля:
  - `category`: код категории товара, строка
  - `category_description`: описание категории товара, строка
  - `probability`: уверенность модели в своем предсказании, дробное число от 0 до 1