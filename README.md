# NU_Datathon_Case_3
Выявление поддельных автомобилей

Date: 14-15 October 2023

### Kaggle 
https://www.kaggle.com/competitions/case3-datasaur-photo/data

### Presentation 
https://www.canva.com/design/DAFxTkZznBU/f76QGPRir3Rouba-tovQsw/view  

### Description
Необходимо обучить модель для выявления поддельных фотографий автотранспорта, так как люди ежегодно пытаются избежать технического осмотра, применяя методы подделки, такие как фотошоп и другие.

В данной задаче по бинарной классификации "0" являются правильным фото автотранспорта, "1" являются фиктивными фото (снятые с экрана монитора, фотошопом и тд.)

В папке "фиктивные" вы можете заметить что фото сами также делятся на подклассы. Все они будут являться "1" для этой задачи. Если ваша модель также правильно будет определять подклассы фиктивных фото, покажите работу модели во время защиты решения задач. За это будет отдельный бонус.

На презентацию кода пройдут топ 7 работ по итогу данного соревнования

### Evaluation (Оценка)
В этом соревновании F1 score используется в качестве основного показателя для ранжирования в таблице лидеров и оценки результатов. F1 score оценивает эффективность классификационных моделей, особенно в задачах бинарной классификации. Он сочетает в себе два важных аспекта производительности модели: точность (Precision) и отзывчивость (Recall).
F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
This metric is especially useful when there is an imbalance in the class distribution, ensuring a balanced consideration of both precision and recall without overemphasizing either.

### Submission File (файл для загрузки)
https://forms.gle/BdsztYAovDtFJbS79 Submission File
You have to submit website link for github repository that contains notebook.ipynb file. Make github repository Public, so we can see the works. Write comprehensive description inside notebook.ipynb
Для каждого file_index в тестовом наборе вы должны предсказать подлинность фотографии, определив класс [0, 1]. Чтобы пояснить, "0" обозначает правильные (подлинные, аутентичные) изображения автомобилей, в то время как "1" представляет вымышленные (сфабрикованные, поддельные) изображения автомобилей. Файл должен содержать заголовок и иметь следующий формат (приведенный ниже формат является лишь примером
file_index,class
76395310,0
78235074,1
74477562,0
70540972,1
73988993,0
75194157,1
77711298,0
72575023,1
75921968,0
79830636,1
etc.
