# engineering_practices_ml
Используется DVC
## Ссылка на хранилище:

https://drive.google.com/drive/folders/17nJZGWKzyW8lqndpPAdX1xSN--inInmR

## Запуск пайплайна:
`dvc repro`

## DAG:
Можно получить с помощью `dvc dag` (также смотри `dvc.yaml`)
Вообще выглядит так:
```
+---------+
| prepare |
+---------+
      *
      *
      *
  +-----+
  | run |
  +-----+
+-------------------------------+
| data/cancer_prec_prec.png.dvc |
+-------------------------------+
+---------------------------------+
| data/cancer_prec_recall.png.dvc |
+---------------------------------+
+-------------------------+
| data/cancer_roc.png.dvc |
+-------------------------+
+------------------------------+
| data/cancer_prec_acc.png.dvc |
+------------------------------+
```
