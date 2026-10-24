# Графики для research paper

Согласованный набор. Все рисунки строятся командой `python -m train.collect --config <config>` и лежат в `final_results/figures/`.
E1 = первый датасет на своём тесте, E2 = первый датасет на втором (внешний), E3 = второй на своём, E4 = второй на первом.
В основной работе: Kvasir-SEG первый, CVC-ClinicDB второй.

| № | Файл | Что показывает | Куда в статье |
|---|---|---|---|
| 1 | `fig01_study_design.png` | схема: датасеты, разбиения, модели, условия E1-E4 | методология |
| 2 | `fig02_dataset_differences.png` | площадь полипа, яркость, разрешение двух датасетов | методология / данные |
| 3 | `fig03_dice_by_condition.png` | Dice по условиям: распределение по изображениям и средние по сидам | результаты (главный) |
| 4 | `fig04_gap_per_seed_dice.png` | разрыв обобщения G по сидам с t-ДИ и порогами 0.05 | результаты (главный) |
| 5 | `fig05_all_metrics.png` | Dice, IoU, precision, recall, HD95, Boundary IoU по условиям | результаты |
| 6 | `fig06_gap_all_metrics.png` | разрыв по всем метрикам с t-ДИ | результаты |
| 7 | `fig07_dice_by_polyp_size.png` | Dice по размеру полипа (small / medium / large) | результаты / анализ причин |
| 8 | `fig08_examples_E2.png`, `fig08_examples_E4.png` | худший, средний и лучший случай на внешнем наборе: изображение, эталон, предсказание, наложение | обсуждение |
| 9 | `fig09_equal_size_comparison.png` | Kvasir-800 против Kvasir-429 и CVC-429 | результаты (равный размер) |
| 10 | `fig10_metric_distributions.png` | распределения Dice, IoU, HD95, Boundary IoU по изображениям | приложение |
| A1 | `figA1_training_curves.png` | loss и Dice по эпохам для каждого сида | приложение |
| A2 | `figA2_delta_sensitivity.png` | вывод правила решения при δ = 0.05, 0.03, 0.10 | приложение |
| A3 | `figA3_error_categories.png` | категории ошибок (появится, если сделана разметка ошибок) | приложение |

Числа лучше давать таблицами, а не рисунками: `tables/metrics_by_condition.csv` (среднее и SD по сидам), `tables/gap_summary.csv` (ДИ), `tables/decision_rule.csv`.
Рисунки 1 и 2 не зависят от результатов обучения.
