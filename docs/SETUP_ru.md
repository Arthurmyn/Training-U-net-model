# Установка на втором ноутбуке (где пойдёт обучение)

Что проверено: **macOS (Apple Silicon), Python 3.9.6, PyTorch 2.8.0 (GPU через MPS)**. Windows, Linux и NVIDIA CUDA я здесь
проверить не мог: для них шаги ниже стандартные, но если что-то не сработает, смотрите раздел «Частые проблемы».

Вся установка занимает 15-30 минут (в основном скачивание). Нужно около 5 ГБ свободного места.

## 0. Что понадобится
- **Python 3.10, 3.11 или 3.12** (3.9 тоже работает; 3.13 пока не рекомендую: у некоторых закреплённых версий нет сборок).
  Проверка: `python --version` (на macOS/Linux может быть `python3`).
- **git**. Проверка: `git --version`.
- Два архива с данными (`Kvasir-seg.zip`, `CVC-ClinicDB.zip`): они не лежат в GitHub, их нужно перенести вручную (флешка, облако).
- Желательно видеокарта NVIDIA (Windows/Linux) или Mac на Apple Silicon. Без видеокарты обучение пойдёт на процессоре и будет в разы медленнее.

## 1. Получить код
```bash
git clone https://github.com/Arthurmyn/Training-U-net-model
cd Training-U-net-model
```
В репозитории уже лежат код, настройки, **зафиксированные списки train/val/test** (`outputs/splits/`) и отчёт аудита.
Переобучать или пересоздавать разбиения не нужно.

## 2. Перенести и проверить данные
Положите оба zip-файла в папку `data/` внутри проекта (создайте её, если нет), затем:

**macOS / Linux**
```bash
mkdir -p data/kvasir_seg data/cvc_clinicdb
(cd data && shasum -a 256 -c data_checksums.sha256)   # обе строки должны быть OK
unzip -q data/Kvasir-seg.zip      -d data/kvasir_seg
unzip -q data/CVC-ClinicDB.zip    -d data/cvc_clinicdb
```
**Windows (PowerShell)**
```powershell
New-Item -ItemType Directory -Force data\kvasir_seg, data\cvc_clinicdb | Out-Null
Get-FileHash -Algorithm SHA256 data\Kvasir-seg.zip, data\CVC-ClinicDB.zip      # сравните с data\data_checksums.sha256
Expand-Archive data\Kvasir-seg.zip   -DestinationPath data\kvasir_seg
Expand-Archive data\CVC-ClinicDB.zip -DestinationPath data\cvc_clinicdb
```
Должно получиться:
```
data/kvasir_seg/Kvasir-SEG/Kvasir-SEG/images   (1000 .jpg)   и  .../masks (1000 .jpg)
data/cvc_clinicdb/PNG/Original                  (612 .png)   и  PNG/Ground Truth (612 .png)
data/cvc_clinicdb/metadata.csv
```
Если контрольные суммы не совпали, файл повреждён при копировании: скопируйте заново. Сырые данные не изменяйте.

## 3. Виртуальное окружение
**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```
**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1          # если запрещено выполнение скриптов: Set-ExecutionPolicy -Scope Process Bypass
python -m pip install --upgrade pip
```
Дальше все команды выполняются **в активированном окружении** (в начале строки видно `(.venv)`).

## 4. Установить PyTorch (выберите ОДИН вариант)
Сначала PyTorch, потом остальное: подходящая сборка зависит от железа.

| Ваше железо | Команда |
|---|---|
| **Mac на Apple Silicon** | `pip install torch` |
| **Windows/Linux + видеокарта NVIDIA** | возьмите точную команду на https://pytorch.org/get-started/locally/ (выберите Stable, ваш ОС, Pip, CUDA). Выглядит так: `pip install torch --index-url https://download.pytorch.org/whl/cu126`; номер CUDA (cu126, cu124 ...) подбирается по драйверу: `nvidia-smi` |
| **Без видеокарты (CPU)** | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |

## 5. Остальные библиотеки
```bash
pip install -r requirements/train.txt
```
Это numpy, pillow, pandas, scipy, matplotlib, pyyaml, tqdm, pytest, jupyter-инструменты. Если нужны **точно те же версии, что у меня**
(проверены вместе), используйте вместо этого `pip install -r requirements/lock.txt` (PyTorch вы уже поставили на шаге 4).

## 6. Проверка установки (по порядку)
```bash
python -c "import torch; print(torch.__version__, 'cuda', torch.cuda.is_available(), 'mps', torch.backends.mps.is_available())"
python -m train.verify_data          # данные те же, что при аудите. Ожидаем: DATA OK
python -m pytest -q                  # ~40 тестов, ~10-60 секунд. Ожидаем: passed
python -m train.run_all --smoke      # мини-обучение на 32 снимках (1-3 мин) + оценка времени полного запуска
```
На первой строке должно быть `cuda True` (NVIDIA) или `mps True` (Mac). Если везде `False`, обучение пойдёт на процессоре.
Smoke-тест в конце печатает оценку: «~N секунд на эпоху; до 100 эпох = X часов на запуск». Запомните её: по ней видно, сколько займёт весь эксперимент.
После теста удалите папку `runs_smoke/`: это не результаты.

Когда все четыре проверки прошли, переходите к `docs/RUNBOOK_ru.md`.

## Частые проблемы
- **`cuda False`, хотя видеокарта есть.** Поставлена CPU-версия PyTorch или не подходит CUDA. Удалите (`pip uninstall torch`) и поставьте по таблице выше, сверяясь с `nvidia-smi`.
- **`CUDA out of memory` / нехватка памяти.** В `train_config.yaml` уменьшите `training.batch_size` (16 -> 8 -> 4) или `model.base_channels` (32 -> 16; но тогда это другая модель: фиксируйте в тексте).
- **Mac: ошибка про неподдерживаемую операцию MPS.** Запустите с `PYTORCH_ENABLE_MPS_FALLBACK=1 python -m train.run_all`.
- **Windows: зависание или ошибки при `num_workers > 0`.** Оставьте `num_workers: 0` (так и стоит).
- **DATA PROBLEM в verify_data.** Файлы отличаются от тех, на которых делался аудит: перераспакуйте архивы (шаг 2) и не смешивайте разные копии датасета.
- **Нет прав / длинные пути на Windows.** Клонируйте в короткий путь, например `C:\work\unet`.
- **Версия из `requirements/lock.txt` не ставится.** Используйте `requirements/train.txt` (без закрепления версий) и запишите версии: `pip freeze`.
