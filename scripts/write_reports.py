"""Build final reports directly from completed run artifacts."""

import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def run(name):
    return json.loads((ROOT / "runs" / name / "result.json").read_text())


warm_names = ["b1-l3001-repeat1", "b1-l3001-repeat2", "b1-l3001-control", "b1-l3001-final-control"]
combined_names = ["combined-l3001-first", "combined-l3001-repeat1", "combined-l3001-repeat2"]
warm = [run(name)["end_to_end_s"] for name in warm_names]
combined = [run(name)["end_to_end_s"] for name in combined_names]
stock_median, best_median = statistics.median(warm), statistics.median(combined)
stock_batch, best_batch = run("b1-fixed-set"), run("combined-fixed-set")
batch_check = json.loads((REPORTS / "combined-fixed-set-comparison.json").read_text())
assert batch_check["all_json_numbers_bit_equal"] and batch_check["all_numeric_prediction_arrays_bit_equal"]
summary = {
    "single_example": {"id": "L3001", "stock_repeat_runs": warm_names, "stock_repeat_times_s": warm,
                       "stock_median_s": stock_median, "optimized_runs": combined_names,
                       "optimized_times_s": combined, "optimized_median_s": best_median,
                       "speedup": stock_median / best_median,
                       "time_reduction_percent": 100 * (1 - best_median / stock_median),
                       "stock_process_peak_mib": run(warm_names[0])["gpu_memory_sampled_peak_mib"],
                       "optimized_process_peak_mib": run(combined_names[0])["gpu_memory_sampled_peak_mib"]},
    "directory_batch": {"examples": 5, "paired_runs": 1, "stock_s": stock_batch["end_to_end_s"],
                        "optimized_s": best_batch["end_to_end_s"], "speedup": batch_check["speedup"],
                        "stock_process_peak_mib": stock_batch["gpu_memory_sampled_peak_mib"],
                        "optimized_process_peak_mib": best_batch["gpu_memory_sampled_peak_mib"],
                        "all_scalar_numbers_bit_equal": True, "all_numeric_prediction_arrays_bit_equal": True,
                        "scalar_fields": len(batch_check["json_metrics"]),
                        "compared_array_fields": len(batch_check["array_comparisons"])},
    "scope": "Performance and exact output equivalence on five fixed public examples; not the full CASP16 scientific benchmark.",
}
(REPORTS / "final-summary.json").write_text(json.dumps(summary, indent=2))
with (REPORTS / "benchmark_results.csv").open("w", newline="") as stream:
    fields = ["label", "end_to_end_s", "gpu_memory_sampled_peak_mib", "device_memory_sampled_peak_mib",
              "host_process_tree_rss_peak_mib", "completion_status"]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for path in sorted((ROOT / "runs").glob("*/result.json")):
        record = json.loads(path.read_text())
        writer.writerow({key: record[key] for key in fields})
with (REPORTS / "output_differences.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["field", "baseline", "optimized", "absolute_difference"])
    for name, row in batch_check["json_metrics"].items():
        writer.writerow([name, row["baseline"], row["candidate"], row["abs_difference"]])

optimization_rows = []
for label, names in [
    ("O1: инициализация с точным RNG", ["o1-l3001-first", "o1-l3001-repeat1", "o1-l3001-repeat2"]),
    ("O1 + O2: сохранение FP32 conditioning", ["o2-l3001-first"]),
    ("O1 + O3: mmap checkpoint", ["o3-l3001-first"]),
    ("O1 + O2 + O3", combined_names),
]:
    values = [run(name)["end_to_end_s"] for name in names]
    optimization_rows.append(f"| {label} | {len(names)} | {statistics.median(values):.2f} |")

report = f"""# Boltz-2 на H200: измеренный результат

Автор исследовательской работы: Арман Аубакиров. Код экспериментов подготовлен с помощью Codex.
Базовая модель и исходная реализация принадлежат авторам Boltz; их авторство не присваивается.

## Результат

| Режим | Штатный B1, с | Оптимизированный, с | Ускорение | Пик процесса, МиБ |
|---|---:|---:|---:|---:|
| Один L3001, медианы повторных новых процессов | {stock_median:.2f} | {best_median:.2f} | {stock_median/best_median:.3f}× | 20 484 → 20 484 |
| Каталог из пяти примеров, одна парная проверка | {stock_batch['end_to_end_s']:.2f} | {best_batch['end_to_end_s']:.2f} | {batch_check['speedup']:.3f}× | {stock_batch['gpu_memory_sampled_peak_mib']:.0f} → {best_batch['gpu_memory_sampled_peak_mib']:.0f} |

Для одного примера время уменьшено на {summary['single_example']['time_reduction_percent']:.2f}%.
Четыре повторных B1: {', '.join(f'{v:.2f}' for v in warm)} с.
Три комбинированных варианта: {', '.join(f'{v:.2f}' for v in combined)} с.
Первый B1 занял {run('b1-l3001-first')['end_to_end_s']:.2f} с и показан отдельно от повторных.
Не следует выдавать самый медленный первый запуск за обычную скорость baseline.

## Большой исходный пример

L3001 использует полный официальный белок L3000 (846 остатков) и официальный лиганд.
B0 без ускоренных ядер полностью завершился за {run('b0-l3001-first')['end_to_end_s']:.2f} с,
наблюдённый пик процесса — {run('b0-l3001-first')['gpu_memory_sampled_peak_mib']:.0f} МиБ.
Обе стадии — структура и affinity — выполнены; OOM не было.
Полученный ColabFold MSA содержит 12 781 строку и совпадает с официальной функцией сборки.
Штатный max_msa_seqs=8192 оставлен без изменения; исходный полученный MSA сохранён полностью.
Точный исторический MSA и конкретный YAML из комментария T4 не опубликованы:
это воспроизведение описанного режима, а не точная историческая репликация.

## Что изменено

1. При строгой загрузке checkpoint не вычисляются временные значения усечённого нормального
   распределения, немедленно перезаписываемые весами. Сохраняются все исходные uniform-вызовы
   SciPy и обычная инициализация Torch. В диагностике эта ненужная работа занимала 36,26 с
   из 51,92 с загрузки двух моделей. Исходный инициализатор восстанавливается после загрузки;
   все пропущенные параметры проверены на принадлежность строго загруженной модели.
2. Неизменные conditioning-тензоры переводятся в FP32 один раз на вызов diffusion,
   вместо повторного преобразования на каждом из 200 шагов. Например, для L3001 один
   такой FP32 тензор занимает 1 176 000 000 байт. Исходный словарь не изменяется,
   кэш не переносится между примерами.
3. Те же локальные checkpoint читаются через torch.load(mmap=True), с прежними параметрами
   map_location и weights_only. Состав, dtype и значения весов не изменяются.

| Вариант | Полных запусков L3001 | Медиана, с |
|---|---:|---:|
{chr(10).join(optimization_rows)}

O2 и O3 по отдельности имеют только пилотные замеры; их индивидуальный вклад нельзя
считать отдельно статистически установленным. Комбинация проверена тремя полными запусками.

## Проверка результатов

В фиксированном наборе L3001, L3003, L3009, L3021 и официальный affinity-пример Boltz
с белком длиной 384 остатка. Последние четыре примера не использовались для выбора оптимизаций.
В пакетной проверке у обоих вариантов получены пять структур, пять confidence-отчётов и
пять affinity-отчётов. Порядок записей сохранён и проверен.

- Все {len(batch_check['json_metrics'])} численных JSON-полей совпали побитно.
- Все численные поля сохранённых массивов совпали побитно; проверено
  {len(batch_check['array_comparisons'])} полей массивов и метаданных.
- Типы, формы и метаданные совпали; NaN/Inf в массивах и координатах CIF отсутствуют.
- На L3001 состояния NumPy и CPU Torch после загрузки обеих полных моделей совпали со stock.
- Дополнительно проверены 12 сочетаний seed/shape/scale/fan и полная загрузка малого checkpoint.
- Допуски зафиксированы до кандидатов в configs/tolerances.json; они не расширялись.

B0 и B1 сами дают различающиеся численные результаты из-за штатных разных путей ядер.
Наши кандидаты сравниваются именно с B1, а не с B0.
Полный CASP16 набор из 140 пар и корреляция 0,65 не проверялись.
Точное совпадение на пяти входах не является доказательством для всех возможных комплексов.

## Измерения и воспроизводимость

Официальный код b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc в upstream/ не менялся.
Веса закреплены на HF revision 6fdef46d763fee7fbb83ca5501ccceff43b85607;
контрольные суммы повторно проверены после экспериментов.
GPU: H200 №2, UUID GPU-6b984734-8ed6-11dd-3b9a-0b6c36ba5e11, MIG выключен.
На карте оставался чужой процесс с 6768 МиБ; его память исключена из пика нашего процесса,
но полный ряд устройства сохранён. Вычислительную конкуренцию отдельно подтвердить не удалось.

E2E включает новый процесс, импорты, загрузку, подготовку входов, все вычисления и запись
результатов. Скачивание ресурсов и последующее сравнение выходов учитываются отдельно.
Никакие готовые предсказания не переиспользовались. Каждый запуск писал новый каталог.
Для набора применён штатный вход-каталог в одном процессе; сумма отдельных CLI-запусков
не выдаётся за пакетную производительность.

VRAM — наблюдённый максимум NVML памяти процесса с шагом около 100 мс, который может
пропустить короткий пик; это не точный allocator peak. Время на общем сервере имеет разброс.
Пакетная скорость получена в одной парной проверке и требует повторов для устойчивой оценки.
RSS — сумма RSS дерева процессов и может многократно учитывать общие страницы.
Документированные научные параметры, seed, checkpoint, MSA, число шагов и samples сохранены.

Результаты: reports/final-summary.json, benchmark_results.csv, output_differences.csv,
combined-fixed-set-comparison.json и каталоги runs/. Полные версии окружения, CLI help,
контрольные суммы и конфиги находятся в manifests/ и configs/.

Источники: https://github.com/jwohlwend/boltz/issues/636 ;
https://predictioncenter.org/casp16/target.cgi?target=L3000&view=sequence ;
https://predictioncenter.org/download_area/CASP16/targets/pharma_ligands/ .
"""
(REPORTS / "OPTIMIZATION_REPORT.md").write_text(report)
(REPORTS / "BASELINE_REPORT.md").write_text(
    "# Исходные измерения Boltz-2 / H200\n\n"
    + f"B0 L3001: {run('b0-l3001-first')['end_to_end_s']:.3f} с; 20704 МиБ; обе стадии завершены.\n"
    + f"B1 первый: {run('b1-l3001-first')['end_to_end_s']:.3f} с; 20484 МиБ.\n"
    + f"B1 повторные: {warm}; медиана {stock_median:.3f} с; 20484 МиБ.\n"
    + f"B1 штатный каталог 5 входов: {stock_batch['end_to_end_s']:.3f} с; {stock_batch['gpu_memory_sampled_peak_mib']:.0f} МиБ.\n\n"
    + "Полные протокол, ограничения и точные проверки — в OPTIMIZATION_REPORT.md.\n"
    + "Источник upstream неизменён; скачивание и MSA подготовлены до замеров.\n"
    + "Первый установочный smoke на upstream prot_custom_msa.yaml имел предупреждение о несовпадении\n"
    + "его MSA и использовал штатную dummy-подстановку. Этот smoke не входит в таблицу ускорения.\n"
    + "L3000 и дополнительный официальный affinity-пример используют отдельно проверенные совпадающие MSA.\n"
)
print(json.dumps(summary, indent=2))
