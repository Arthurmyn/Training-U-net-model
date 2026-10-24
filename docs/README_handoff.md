# Что в архиве

| Файл / папка | Содержимое |
|---|---|
| `results/per_image_metrics.csv` | **общий файл**: направление, датасеты, условие, сид, изображение, Dice, площади, последовательность |
| `runs/*/per_image_dice.csv` | то же по запускам (6 запусков) |
| `runs/*/metrics.csv` | логи обучения по эпохам (train/val loss, train/val Dice, lr, секунды) |
| `runs/*/run.json`, `eval_summary.json` | настройки, версии, GPU, лучшая эпоха, средние Dice |
| `runs/run_all.log` | общий лог запуска |
| `config.yaml`, `train_config.yaml` | настройки аудита и обучения |
| `outputs/audit_report.md`, `summary.json` | отчёт аудита |
| `outputs/splits/` | зафиксированные разбиения и списки внешнего теста |
| `results/summary.md`, `results/tables/`, `results/fig_dice_distributions.png` | статистика и график |
| `docs/research_log.md`, `docs/errors_log.md` | хронология решений и список ошибок |

**Важно:** запуски на CVC-ClinicDB недообучены (см. `docs/errors_log.md`, E10); результаты по CVC предварительные.
Веса моделей (`best.pt`) в архив не включены.
