1st place solution to Mercari Price Suggestion Challenge
========================================================

https://www.kaggle.com/c/mercari-price-suggestion-challenge/

Presentation link (talk given on 31 March 2018)
========================================================

https://github.com/pjankiewicz/mercari-solution/raw/master/presentation/build/yandex.pdf

Model stats
-----------

```
merge_predictions =
-0.0203
+0.0604 * data1_huber
+0.1051 * data1_huber
+0.0911 * data1_clf
+0.0760 * data1_clf
+0.0851 * data2_huber_bin
+0.0981 * data2_huber
+0.0819 * data2_clf_bin
+0.0717 * data2_clf
+0.0958 * data3_huber_bin
+0.1226 * data3_huber
+0.0578 * data3_clf_bin
+0.0642 * data3_clf
=> RMSLE 0.3733
```
