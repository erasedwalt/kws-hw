# KWS HW

Реализовал стриминг, сжал и ускорил модель: самая маленькая модель меньше базовой в `~27` раз и быстрее в `~14` раз, если считать в MACs
(это `Dynamic Quantization + Attention Distillation + Dark Knowledge Distillation`). Её я учил ~120 эпох. Именно поэтому хочется также выделить связку `Dynamic Quantization + Structured Pruning + Attention Distillation + Dark Knowledge Distillation`, которая училась только 40 эпох до нужного результата, и при этом она в `~21` раз меньше и в `~10` раз быстрее базовой модели.

Логи принтовал прямо внутри ноутбука. Ячейки для скачивания весов соответствующих моделей тоже есть (они рядом с запусками тренировок). Нет таких ячеек только для квантизованных моделей, они получаются из своих неквантизованных версий одной строкой кода за 0 секунд (в ноутбуке эти строки есть).

По идее, воспроизвести модели можно аккуратным протыкиванием всего ноутбука (seed я фиксировал, тренировать нужно столько, сколько тренировал я).

## Стриминг

Реализовал следующую идею для стриминга. Есть параметр `max_buffer_len` — столько запоминаем фреймов. А конкретно для каждого фрейма модель
запоминает энергии в attention (то есть до применения softmax'а) и скрытые состояния GRU, пока не заполнится буфер. Соответственно,
при предсказании вероятности для очередного чанка участвуют скрытые состояния GRU из прошлых `max_buffer_len` фреймов.

Для адекватной работы с этими фреймами я вычислил, что `1 секунда ~= 100 степов в мелспеке` и `1 секунда ~= 11-12 фреймов внутри модели (после свертки)`.
Поэтому поставил по умолчанию `max_buffer_len = 24`, то есть примерно 2 секунды.

Чтобы проверить на работоспособность, взял запись с ключевым словом, скачал запись речи из LJSpeech, сконкатенировал речь спереди и сзади к записи с ключевым словом
и прогнал на этом всём модель.

Получилось так:

![Streaming](https://github.com/erasedwalt/kws-hw/blob/main/pictures/streaming.png)

В принципе, как и ожидалось: нулевая вероятность вначале, затем в середине появляется ключевое слово, модель его детектит, а затем вероятность снова затухает.

## Сжатие и ускорение

- Base. `MACs: 1952368, Size (KB): 278.51, Metric: 2.6e-5`.
- Dark Knowledge Distillation. `MACs: 305167, Size (KB): 36.07, Metric: 4.59e-5`. Кажется, подсмотрел, как именно используются `alpha` и `T` в дистилляции, вот [тут](https://github.com/peterliht/knowledge-distillation-pytorch/tree/master). Ну и после этого было всё просто, я просто сжал модель, уменьшив количество каналов в свертке в 2 раза, ядро свертки по одной оси до 15-ти и в 3 раза количество каналов в GRU. Нужного результата пришлось ждать около 70 эпох.
- Dark Knowledge Distillation + Dynamic Quantization. `MACs: 305167, Size (KB): 16.72, Metric: 4.61e-5`. В туториалах торча посмотрел, как динамически квантизовать модель, сделал.
- Dark Knowledge Distillation 2. `MACs: 186924, Size (KB): 18.76, Metric: 4.77e-05`. Показалось, что этим методом можно ещё сильнее сжать модель. Уменьшил количество каналов в свертке в 2 раза, ядро свертки по одной размерности до 13-ти и размер скрытого слоя в GRU в 5 раз. Это всё по сравнению с базововй моделью. Пришлось ждать нужного результата около 175 эпох.
- Dark Knowledge Distillation 2 + Dynamic Quantization. `MACs: 186924, Size (KB): 12.16, Metric: 4.72e-5`. Применил квантизацию к предыдущей модели.
- Attention Distillation + Dark Knowledge Distillation. `MACs: 137698, Size (KB): 13.14, Metric: 5.28e-05`. Уменьшил теперь количество каналов в свертке в 3 раза, ядро оставил неизменным (это нужно для дистилляции attention), количество каналов в GRU уменьшил в 6 раз. Слепил как-то лосс по примеру простой дистилляции, ну и нельзя сказать, что не заработало. Пришлось ждать нужного результата около 120 эпох.
- Attention Distillation + Dark Knowledge Distillation + Dynamic Quantization. `MACs: 137698, Size (KB): 10.28, Metric: 5.21e-05`. Просто квантизовал модель из предыдущей стадии. Эту модель получилось ускорить и сжать лучше всего. Она меньше базовой в `~27` раз и быстрее в `~14` раз (если считать в MACs).
- Base + Dynamic Quantization. `MACs: 1952368, Size (KB): 80.6, Metric: 2.74e-5`. Применил квантизацию к базовой модели.
- Попробовал ещё заиспользовать Static Quantization и Quantization Aware Training, но, как я понял, torch не поддерживает такие типы квантизаций для GRU.
- Structural Pruning + Attention Distillation + Dark Knowledge Distillation. `MACs: 193420, Size (KB): 22.2, Metric: 4.77e-05`. Если я правильно понял идею структурного прунинга, то мы хотим занулять целые фильтры. За этой целью я разделил некоторые модули модели на две части, а именно свертку, гру и линейный слой (надеюсь, я не испортил таким образом топологию модели). На одну из двух частей я наложил огромную регуляризацию, из-за чего веса на них почти занулились. Обучил так модель. Затем я написал аналогичный класс модели, но уже без этих слоев с нулевыми весами, загрузил в неё уже обученные ненулевые веса. И тадааам, всё заработало!
- Dynamic Quantization + Structural Pruning + Attention Distillation + Dark Knowledge Distillation. `MACs: 193420, Size (KB): 13.03, Metric: 4.85e-05`. Квантизовал предыдущую модель.

### Графики

_На графиках KD — Knowledge Distillation, AD — Attention Distillation, DQ — Dynamic Quantization, SP — Structural Pruning, B — Base._

![Quantization](https://github.com/erasedwalt/kws-hw/blob/main/pictures/quantization.png)

Тут можно посмотреть, как квантизация влияет на размер модели в зависимости от того, насколько большой у неё изначально размер. Любопытно. Получается, что уже небольшие модели сжимаются не так сильно.

Также можно разглядеть, что качество не всегда падает, что тоже любопытно. При этом качество меняется в пределах разумного, и в то же время размер уменьшается сильно. Поэтому вывод — квантизация класс.

![Distillation](https://github.com/erasedwalt/kws-hw/blob/main/pictures/distillation.png)

Здесь попытался выделить влияние дистилляции. Видно, что огромный вклад в сжатие сделала именно дистилляция.

На самом деле, был очень приятно удивлен таким хорошим результатам дистилляции. Вариант намного более долгий по сравнению с квантизацией (нужно много эпох), но буст очень хороший. Ещё больше мастхэв.

![Size vs Epochs](https://github.com/erasedwalt/kws-hw/blob/main/pictures/size%20vs%20epochs.png)

Здесь можно разглядеть, как дополнительный дистилляционный лосс от аттеншна и структурный прунинг позволяют сжать модель быстрее. В случае с прунингом это видимо потому, что изначально мы минимизируем более сложный функционал, поэтому там больше свободы у оптимизатора, хоть и сильные ограничения в виде регуляризации. А в случае с аттеншн совсем понятно: там дополнительно текут градиенты, что всегда хорошо, и всё ускоряет. Вывод — SP и AD топ.

![MACs vs Size](https://github.com/erasedwalt/kws-hw/blob/main/pictures/size%20vs%20macs.png)

Тут видимо можно сделать вывод, что размер и MACs модели ведут себя примерно пропорционально, только с большим коэффициентом, что логично.

## Ссылки

Ссылки на модели.

- [B](https://drive.google.com/file/d/12FHzJRLIWxq5bQFa9IDOEuqFfkz1q5Ic/view?usp=sharing)
- [DQ+B](https://drive.google.com/file/d/1-CnggSb4h6iNamOHsAWdIn8dZCwdBDOW/view?usp=sharing)
- [KD](https://drive.google.com/file/d/1-02tWOG3fwVjWRU_qthLXCWtUKqUhJtT/view?usp=sharing)
- [DQ+KD](https://drive.google.com/file/d/1-DdVoVKouoejeREM93MbHjHGast_Sn0a/view?usp=sharing)
- [KD2](https://drive.google.com/file/d/1-1DHkuwd8_NlNB4SOH7lcz12LANwCQ-K/view?usp=sharing)
- [DQ+KD2](https://drive.google.com/file/d/1-4wzaS5IuUokdFaRV2mDqws9fXj2C3Ce/view?usp=sharing)
- [AD+KD](https://drive.google.com/file/d/1xVPynAedL4JYmniC_YSpccWyeJtDKUxG/view?usp=sharing)
- [DQ+AD+KD (Best Result)](https://drive.google.com/file/d/1-1Hs45mFNO8pHjI2Ge8m8T_fwmgzMZ23/view?usp=sharing)
- [SP+AD+KD (post) (Favourite in my heart)](https://drive.google.com/file/d/1-6cscMhyPT71J73ToGrz4ahAVxJXtYID/view?usp=sharing)
- [DQ+SP+AD+KD](https://drive.google.com/file/d/1-C7vXNvEumdktsIpYGbnIASjiymvYgQ6/view?usp=sharing)
