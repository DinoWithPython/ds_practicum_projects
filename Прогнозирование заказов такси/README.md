# Прогнозирование заказов такси

<br/>
<table>
    <tr>
        <td><b>Название проекта (+ссылка)</b></td>
        <td><b>Тема проекта и задача проекта</b></td>
        <td><b>Используемые инструменты</b></td>
        <td><b>Темы инф. материалов и рекомендации ревьювера</b></td>
    </tr>
    <tr>
        <td><a href="https://github.com/DinoWithPython/ds_practicum_projects/blob/main/Прогнозирование%20заказов%20такси/09%20Прогнозирование%20заказов%20такси.ipynb" target="_blank"><b>"Прогнозирование заказов такси."</b></a></td>
        <td><b>Временные ряды.</b> Чтобы привлекать больше водителей в период пиковой нагрузки, нужно спрогнозировать количество заказов такси на следующий час. Постройте модель для такого предсказания.</td>
        <td><b>LGBMRegressor, seasonal_decompose, TimeSeriesSplit </b></td>
        <td>Ссылки на лекции по временным рядам, чтобы лучше разобраться в теме.</td>
    </tr>
</table>

## Общий вывод и рекомендации заказчику
Компания «Чётенькое такси» собрала исторические данные о заказах такси в аэропортах. Чтобы привлекать больше водителей в период пиковой нагрузки, нужно спрогнозировать количество заказов такси на следующий час. Построить модель для такого предсказания.    


После использования ресемплирования, скользящего среднего(размер окна 24), выделения анализируемого периода до двух недель, можно сделать выводы:
* По тренду видим, что общее количество заказов за это время растет;    
* Так же заметно, что количество заказов по утрам меньше и растет в течение дня;
* Ряд не является стационарным, поскольку его распределение со временем меняется.

Необходимости в предобработке как такой не было.

В качестве основной модели для предсказания такси на следующий час при помощи `GridSearchCV` выбрана `LGBMRegressor`('learning_rate'=0.1, 'num_leaves'=50). Модель показала хорошие рузультаты метрики RMSE: 26.07.


#### 📖 **Полезные ссылки**

* [Полезная лекция про временные ряды](https://www.youtube.com/watch?v=u433nrxdf5k)
* [Б.Б. Демешев - временные ряды](https://disk.yandex.ru/i/LiDHB-B3A6Lz5A)
* [Базовое применение ARIMA](https://colab.research.google.com/drive/17RnG91Eq8JBKyxToNzvCvjibfxum-oPj?usp=sharing)
* [Канторович - Анализ временных рядов](https://yadi.sk/i/IOkUOS3hTXf3gg)
* [По теме выше](https://facebook.github.io/prophet/)
* [Ссылка раз](https://facebook.github.io/prophet/docs/quick_start.html#python-api)
* [Ссылка два](https://nbviewer.jupyter.org/github/miptgirl/habra_materials/blob/master/prophet/habra_data.ipynb)
