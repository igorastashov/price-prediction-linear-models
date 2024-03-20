##  Предсказание стоимости автомобиля на основе разработанной линейной модели и реализация FastAPI веб-сервиса для презентации решения


Асташов И.В., Юсупов Ш.Ш., 2024.

Репозиторий содержит проект с первичной предобработкой полученных данных, их разведочным анализом
и обученной моделью, предсказывающую стоимость автомобиля по ее характеристикам. 

Проект выполнен в рамках курса «Машинное обучение» магистерской программы НИУ ВШЭ 
[«Машинное обучение и высоконагруженные системы»](https://www.hse.ru/ma/mlds/).

**Для запуска предобученной модели на ваших данных, смотри [(6) Быстрый старт](https://github.com/igorastashov/price-prediction-linear-models/tree/dev?tab=readme-ov-file#6-%D0%B1%D1%8B%D1%81%D1%82%D1%80%D1%8B%D0%B9-%D1%81%D1%82%D0%B0%D1%80%D1%82).** </br>
**Обработка первичных данных и разработка модели: [notebooks/homework_practice_01_Astashov.ipynb](https://github.com/igorastashov/price-prediction-linear-models/blob/dev/notebooks/homework_practice_01_Astashov.ipynb).**

## (1) Задача

Разработать линейную модель для предсказания стоимости автомобиля по предоставленным данным, и реализовать FastAPI веб-сервис
для презентации решения.

Для оценки качества модели использовать метрики: `MSE` и `r2`. Бизнес-метрика: доля предсказаний, отличающихся
от реальных цен не более чем на 10% по обе стороны.

## (2) Технологии

- Python
- Sklearn
- Statsmodels
- FastAPI
- Redis
- Uvicorn
- PostgreSQL

## (3) Файлы

- **app/db/models/models.py**: структура таблицы (модель ORM);
- **app/db/schemas/schema.py**: схема валидации данных;
- **app/db/sessions/sessions.py**: реализация операций CRUD для работы с объектами Car;
- **app/db.py**: подготовка и настройка среды для работы с базой данных PostgreSQL;
- **app/ds/pre_processing.py**: файл c классом для предобработки данных;
- **app/ds/ridge.py**: модель;
- **app/models/download_model.sh**: скрипт для загрузки предобученной модели `model_ridge.pkl`;
- **app/weights/download_transformers.sh**: скрипт для загрузки `.pkl` файлов с числовыми значениями необходимые для инференса;
- **app/main.py**: файл приложения FastAPI;
- **data/download_data.sh**: скрипт для загрузки `cars_train.csv`, `cars_test.csv` и `cars_test_cut.csv`;
- **notebooks/homework_practice_01_Astashov.ipynb**: предобработка данных, разведочный анализ, поиск наилучшей модели; 
- **Dockerfile**: файл с инструкцией для создания Docker образа;
- **requirements.txt**: файл зависимостей.

## (4) Запуск локально

### Shell

```
# Загрузка модели
cd app/models
bash download_model.sh
cd ../..
```

```
# Загрузка трансформеров
cd app/weights
bash download_transformers.sh
cd ../..
```

```
$ python -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
$ cd app
$ uvicorn main:app --reload
```

### Docker

```
$ docker build . -t fastapi_app:latest
$ docker run -it --rm -p '8000:8000' fastapi_app
```
Открыть документацию API для просмотра: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## (5) Данные

Загрузка данных:

```
# Загрузка cars_train.csv, cars_test.csv и cars_test_cut.csv
cd data
bash download_data.sh
cd ../..
```

Данные об автомобилях `data/*.csv`:

- `cars_train.csv`: тренировочный набор данных;
- `cars_test.csv`: тестовый набор данных;
- `cars_test_cut.csv`: урезанный тестовый набор данных.

Файлы включают такие столбцы как:
- **name**: марка и модель авто;
- **year**: год выпуска;
- **selling_price**: стоимость авто (только для train);
- **km_driven**: общий пробег;
- **fuel**: тип топлива;
- **seller_type**: тип продавца;
- **transmission**: тип коробки передач;
- **owner**: число владельцев;
- **mileage**: расход топлива;
- **engine**: объем двигателя;
- **max_power**: мощность двигателя;
- **torque**: крутящий момент;
- **seats**: число мест.

## (6) Быстрый старт

![docker-run](https://github.com/igorastashov/price-prediction-linear-models/assets/90093310/82addfa1-68b4-43cf-a422-beb33a0a6f20)

### Предсказание на одном объекте

- Используя Swagger UI для отправки запроса к сервису, необходимо выполнить пункт [(4) Запуск локально](https://github.com/igorastashov/price-prediction-linear-models/tree/dev?tab=readme-ov-file#4-%D0%B7%D0%B0%D0%BF%D1%83%D1%81%D0%BA-%D0%BB%D0%BE%D0%BA%D0%B0%D0%BB%D1%8C%D0%BD%D0%BE), либо воспользоваться приложением на [Render]();
- Добавить в базу данных информацию о новом автомобиле путем ввода в форму для отправки запросов `/add_car` необходимые характеристики автомобиля;
- Используя форму `/predict_by_range`, указать в графе `Start PK value` и `End PK value` уникальный ключ добавленного в базу данных автомобиля.

![predict-item](https://github.com/igorastashov/price-prediction-linear-models/assets/90093310/441fd963-b7cb-4396-8b13-a2f3091b1295)

### Предсказание на нескольких объектах

- По аналогии с [предсказанием на одном объекте](https://github.com/igorastashov/price-prediction-linear-models/tree/dev?tab=readme-ov-file#%D0%BF%D1%80%D0%B5%D0%B4%D1%81%D0%BA%D0%B0%D0%B7%D0%B0%D0%BD%D0%B8%D0%B5-%D0%BD%D0%B0-%D0%BE%D0%B4%D0%BD%D0%BE%D0%BC-%D0%BE%D0%B1%D1%8A%D0%B5%D0%BA%D1%82%D0%B5), загрузить в форму `/add_cars_from_csv` `.csv` файл, с целью добавления автомобилей в базу данных.
Который будет содержать данные, соответствующие [(8) Схеме Car](https://github.com/igorastashov/price-prediction-linear-models/tree/dev?tab=readme-ov-file#8-%D1%81%D1%85%D0%B5%D0%BC%D0%B0);
- Используя форму `/predict_by_range`, указать в графе `Start PK value` и `End PK value` уникальные ключи добавленных в базу данных автомобилей.

![predict-items](https://github.com/igorastashov/price-prediction-linear-models/assets/90093310/b80f128f-1d93-44c1-87af-7616445001fb)

## (7) Обучение и оценка модели

### Обучение и оценка модели

- Необходимо выполнить пункт [(4) Запуск локально](https://github.com/igorastashov/price-prediction-linear-models/tree/dev?tab=readme-ov-file#4-%D0%B7%D0%B0%D0%BF%D1%83%D1%81%D0%BA-%D0%BB%D0%BE%D0%BA%D0%B0%D0%BB%D1%8C%D0%BD%D0%BE), либо воспользоваться приложением на [Render]().
- Загрузить в Swagger UI файл `.csv` в форму `/fit_model`. Который будет содержать данные, соответствующие [(8) Схеме Car](https://github.com/igorastashov/price-prediction-linear-models/tree/dev?tab=readme-ov-file#8-%D1%81%D1%85%D0%B5%D0%BC%D0%B0), с добавленным столбцом `selling_price: int`.

![fit-model](https://github.com/igorastashov/price-prediction-linear-models/assets/90093310/6d92431f-4a82-40c1-b216-cb17e1a72e1b)

## (8) Схемы

Car:
- pk: int
- name: str
- year: int
- km_driven: int
- fuel: str
- seller_type: str
- transmission: str
- owner: str
- mileage: str
- engine: str
- max_power: str
- torque: str
- seats: float

Predictions:
- car_pk: int
- predicted_price: float

## (9) Дизайн API

**/**
- GET: Базовая информация о приложении

**/add_car**
- POST: Добавить новый автомобиль

**/add_cars_from_csv**
- POST: Добавить новые автомобили из .csv файла

**/cars**
- GET: Вывести список автомобилей в соответствии с выбранными характеристиками

**/car/{pk}**
- GET: Вывести автомобиль по указанному ключу

**/car/{pk}**
- PATCH: Обновить информацию об автомобиле по его уникальному ключу

**/predict_by_range/**
- GET: Предсказать стоимость автомобиля/-ей по выбранному диапазону ключей

**/fit_model**
- POST: Переобучить модель на новых данных

## (10) Описание  работы

### Работа над проектом включает:
1. Разведочный анализ данных;
2. Подбор модели и ее гиперпараметров;
3. Обработка признаков и создание новых;
4. Разработка веб-сервиса.


### 1. Разведочный анализ:

- анализ парной корреляции признаков и целевой переменной;
- построение диаграмм рассеяния;
- построение графиков распределения;
- анализ выбросов.

Краткий вывод по результатам анализа:
- высокая корреляция стоимости авто с мощностью двигателя, годом выпуска и пробегом;
- корреляция между годом выпуска и мощностью двигателя низкая. И на оборот, высокая корреляция между годом выпуска и пробегом;
- большое число выбросов количественных признаков и целевой переменной. Распределение `max_torque_rpm` - мультимодально.

### 2. Подбор модели и ее гиперпараметров

Обучены модели:
- LinearRegression;
- Lasso;
- ElasticNet;
- Ridge.

Лучший результат на тренировочных и тестовых данных по целевым метрикам показала модель Ridge, где r2 на тестовых данных `0.626`.

### 3. Обработка признаков и создание новых

На основании проведенного анализа произведено:
- удаление дубликатов;
- преобразование признаков к необходимому типу данных;
- обработка признака `torque` - разделение на два признака `toque` и `max_torque_rpm`;
- замена пропусков медианным значением;
- замена выбросов граничными значениями, определенными по квантилям;
- кодирование категориальных переменных методом One Hot Encoding;
- создание полиномиальных признаков;
- стандартизация данных;
- логарифмирование признаков и целевой переменной.

В результате Ridge модель показала с преобразованными данными r2 на тесте `0.886`.

С целью увеличения качества модели, хотелось бы проверить на сколько улучшится r2 при получении марки автомобиля по ее названию. Так же необходимо провести эксперименты 
по подбору оптимального кодировщика категориальных переменных, попробовать заполнить пропуски предсказанными значениями, поработать отдельно с выбросами,
добавив новые категории.

### 4. Разработка веб-сервиса.
Разработан веб-сервис, [(6) Быстрый старт](https://github.com/igorastashov/price-prediction-linear-models/tree/dev?tab=readme-ov-file#6-%D0%B1%D1%8B%D1%81%D1%82%D1%80%D1%8B%D0%B9-%D1%81%D1%82%D0%B0%D1%80%D1%82).

## Благодарности

Используемые материалы: 
- [Лекции ВШЭ по ML](https://www.youtube.com/playlist?list=PLmA-1xX7IuzCglOyTkTZ_bBHKd8eUr8pC);
- [Лекции ВШЭ по Applied Python](https://www.youtube.com/playlist?list=PLmA-1xX7IuzADGz3hSgPPm6ib11Z0HSML);
- [Лекции ВШЭ по PrePromProg](https://www.youtube.com/playlist?list=PLmA-1xX7IuzBaM_2Mi5AeNBUEOZgjsy48);
- [Артем Шумейко](https://www.youtube.com/watch?v=fm4LTvMyiwE).
