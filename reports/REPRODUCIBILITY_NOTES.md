# Повторный запуск

Рабочий проект: `/mnt/data/arman-projects/boltz-h200`.
Окружение и все ресурсы уже установлены. Каждый новый запуск требует нового имени:

```bash
cd /mnt/data/arman-projects/boltz-h200
.venv/bin/python scripts/run_benchmark.py new-stock --input L3001
.venv/bin/python scripts/run_benchmark.py new-optimized --input L3001 \
  --launcher /mnt/data/arman-projects/boltz-h200/scripts/optimized_cli.py
.venv/bin/python scripts/compare_outputs.py new-stock new-optimized
.venv/bin/python scripts/validate_outputs.py new-optimized
```

Для штатного пакетного режима используйте `--input fixed_set` в обоих запусках.
Это один процесс на каталог из пяти YAML, а не цикл отдельных CLI-процессов.
Скрипт сам закрепляет физическую GPU №2 по UUID до запуска CUDA-процесса.

На чистом месте требуются Python 3.12, совместимый драйвер NVIDIA и H200.
Сначала восстановите исходники и зависимости:

```bash
git clone https://github.com/jwohlwend/boltz.git upstream
git -C upstream checkout --detach b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc
python3 -m venv .venv
.venv/bin/python -m pip install -r configs/requirements-locked.txt \
  --extra-index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install --no-deps -e ./upstream
mkdir -p cache data/raw data/inputs data/msa runs reports
.venv/bin/python scripts/prepare_assets.py inputs
.venv/bin/python scripts/prepare_assets.py models
.venv/bin/python scripts/prepare_assets.py msa
```

Frozen L3000 MSA и исходные CASP файлы сохранены в локальной истории проекта.
Если MSA уже существует, подготовка сохраняет его. Повторная генерация на сервере
ColabFold может вернуть другой снимок базы; это новый эксперимент, а не точная
репликация. Нужный SHA256: `468f207bb760a39eece990301aa27b9456cda14e861c4668756457882ff8c578`.

Полные версии находятся в `configs/requirements-locked.txt` и
`manifests/pip-freeze.txt`; коммит Boltz, checkpoint settings и CLI defaults —
в `manifests/` и `configs/`. CUDA runtime PyTorch — 12.8, драйвер измерений — 580.82.07.
Каталоги результатов новые, скачивание и MSA находятся вне E2E обоих вариантов.
Все импорты, загрузка, вычисления и запись результатов входят в E2E.

Первый установочный smoke обнаружил несовпадение MSA с последовательностью в
upstream `prot_custom_msa.yaml`. Его штатная dummy-подстановка не использовалась
в измерениях L3000 и фиксированного набора; их MSA отдельно проверены.
Предупреждения Lightning о версии checkpoint, `fork`, Tensor Cores и устаревшем
TorchScript не были ошибками завершённых запусков. Версии не обновлялись между ними.

Оптимизации ограничены prediction CLI и строгой загрузкой checkpoint.
Случайный инициализатор восстанавливается после загрузки; обучение и частичная
загрузка checkpoint этим экспериментом не валидировались.
На момент исходного отчёта изменения не публиковались. Эти исследовательские артефакты позднее опубликованы в отдельном репозитории; патч в upstream не отправлялся.
