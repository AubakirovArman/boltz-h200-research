"""Write a result report only after complete, finite coverage of all 140 pairs."""

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

from score import score


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def main(run_root):
    summary = score(run_root, 2500)
    assert summary["coverage"]["complete"], "Do not write a final report from partial predictions"
    assert {key: value["n"] for key, value in summary["per_target"].items()} == {"L1000": 17, "L3000": 123}
    destination = ROOT / "reports/casp16-issue636"
    destination.mkdir(parents=True, exist_ok=True)
    validation = {}
    checksums = {}
    for target, expected in [("L1000", 17), ("L3000", 123)]:
        run = run_root / target
        result = json.loads((run / "result.json").read_text())
        assert result["returncode"] == 0
        confidence = list((run / "output").rglob("confidence_*.json"))
        affinity = list((run / "output").rglob("affinity_*.json"))
        assert len({path.parent.name for path in confidence}) == expected
        assert len(confidence) == expected * 5
        assert len(affinity) == expected
        for path in [*confidence, *affinity]:
            checksums[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        validation[target] = {"complete_pairs": expected, "confidence_files": len(confidence),
                              "affinity_files": len(affinity), "source_config": str((run / "config.json").relative_to(ROOT))}
        shutil.copy2(run / "config.json", destination / (target + "-config.json"))
        for attempt_config in run.glob("config.attempt*.json"):
            shutil.copy2(attempt_config, destination / (target + "-" + attempt_config.name))
        shutil.copy2(run / "result.json", destination / (target + "-run-result.json"))
    (destination / "output-validation.json").write_text(json.dumps(validation, indent=2))
    (destination / "output-checksums.json").write_text(json.dumps(checksums, indent=2))
    shutil.copy2(run_root / "scoring-summary.json", destination / "scoring-summary.json")
    shutil.copy2(run_root / "predictions-and-labels.csv", destination / "predictions-and-labels.csv")
    aggregate = summary["aggregate"]
    raw_uniform = aggregate["raw"]["uniform_target_mean"]
    raw_weighted = aggregate["raw"]["compound_weighted_target_mean"]
    corrected_uniform = aggregate["mw_corrected"]["uniform_target_mean"]
    corrected_weighted = aggregate["mw_corrected"]["compound_weighted_target_mean"]
    rows = []
    for target, details in summary["per_target"].items():
        metrics = details["metrics"]
        rows.append(f"| {target} | {details['n']} | {metrics['raw']['pearson_r']:.6f} | {metrics['mw_corrected']['pearson_r']:.6f} | {metrics['mw_only']['pearson_r']:.6f} |")
    ci = summary["bootstrapping"]["ci95"]["mw_corrected/compound_weighted"]
    effect = summary["bootstrapping"]["mw_effect_weighted_ci95"]
    report = f'''# Boltz-2: расследование расхождения CASP16 в issue #636

## Вывод

Подтверждены два существенных источника расхождения протоколов: обычное среднее вместо весов по числу соединений и отсутствие штатной MW-поправки в CLI по умолчанию. На одних и тех же 140 прогнозах они дают переход **{raw_uniform:.6f} → {raw_weighted:.6f} → {corrected_weighted:.6f}**. Это объясняет основную часть разрыва, но ровно 0.65 не воспроизведено: остаётся {0.65 - corrected_weighted:.6f}. Причина этого остатка не установлена; для точного сопоставления нужны исходные входы и прогнозы авторов. Возможное влияние seed, MSA и различий входных молекул отдельно не измерено.

## Полная независимая проверка

Выполнены все **140 пар**: 17 L1000 и 123 L3000. Пропусков нет. Для каждой пары получены пять структурных confidence-отчётов и один affinity-отчёт. Численные affinity-выходы конечны. Официальный исходный код не изменялся; оптимизационные обёртки не применялись.

| Белок | Пар | Raw ensemble R | С официальной MW-поправкой R | Контроль: только MW |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

| Режим расчёта на тех же прогнозах | Pearson R |
|---|---:|
| Raw, обычное среднее двух белков: метод из issue | {raw_uniform:.6f} |
| Raw, вес каждого белка равен числу соединений | {raw_weighted:.6f} |
| MW-поправка, обычное среднее двух белков | {corrected_uniform:.6f} |
| MW-поправка, вес по числу соединений | **{corrected_weighted:.6f}** |

Первый переход, только агрегация: {raw_weighted - raw_uniform:+.6f}.
Второй переход, фиксированная поправка при той же взвешенной метрике: {corrected_weighted - raw_weighted:+.6f}.
Общий переход от raw/обычного среднего к MW/взвешенному: {corrected_weighted - raw_uniform:+.6f}.

95% percentile-интервал целевого MW/взвешенного результата при 2500 парных стратифицированных bootstrap-повторах: [{ci[0]:.6f}, {ci[1]:.6f}].
Интервал изменения от MW-поправки при взвешенной метрике: [{effect[0]:+.6f}, {effect[1]:+.6f}]. Это оценка неопределённости на текущей выборке, а не доказательство совпадения с неизвестными исходными прогнозами авторов статьи.

## Подтверждённые различия протокола

**Агрегация.** В issue обычное среднее R двух белков, тогда как §5 и D.2.2 статьи описывают веса по числу соединений. Для опубликованных округлённых 0.44 и 0.59 это 0.515 против (17×0.44+123×0.59)/140=0.5717857. Одно это изменение не даёт 0.65.

**Молекулярная масса.** B.5.2 статьи описывает скорректированный ансамбль. В закреплённом коде класса параметр по умолчанию True, но CLI передаёт False без `--affinity_mw_correction`. Мы применили формулу официального кода: `1.03525938 * raw - 0.59992683 * MolWt**0.3 + 2.83288489`. Коэффициенты не подбирались на CASP16. Эта поправка зависит от молекулы, поэтому, в отличие от обычного сдвига среднего, может менять корреляцию.

В формуле B.5.2 статьи показана MW, а в опубликованном коде используется `MolWt**0.3`. Для воспроизводимости здесь зафиксирован именно опубликованный код с его коэффициентами; формула не восстанавливалась подбором по тестовым меткам.

**Единицы и центрирование.** Переход от log10(IC50 в µM) к официальным энергетическим значениям аффинен; на обоих CSV корреляция этих двух меток практически 1. Сдвиг среднего внутри одного белка также сохраняет Pearson R. Эти действия не являются объяснением разрыва.

## Входные неоднозначности

У CSV экспериментальных меток и исходных challenge TSV обнаружены 19 химических различий: 17 по стереохимии и 2 по графу молекулы (L3027, L3101). Для первичного эксперимента источник не смешивался: использованы SMILES из текущих affinity CSV, один полный белок и один основной лиганд. Их влияние на разницу со статьёй не объявляется измеренным без отдельного контролируемого сравнения. См. `research/casp16/input-structure-audit.json`.

В obsolete-файле L1000 есть два разных assay-столбца IC50; актуальный CSV оставляет CHYMASE_IC50_COLI_NH. В основном анализе использован именно актуальный столбец. Не выбирался тот assay, который даёт больший R.

## Воспроизводимость и предел вывода

План четырёх сравнений был зафиксирован в локальном commit 97112ed до завершения полного расчёта. Исходный commit Boltz: b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc. Веса закреплены в manifests/model-revision.json; оба замороженных MSA сохранены. Число циклов/шагов/samples взято из issue: структура 10/200/5, affinity штатно 5/200/5; seed=0 на процесс. GPU: физическая H200 №2. Две целевые серии выполнялись в разных процессах на этой же карте; их время не является сравнением производительности. Терминальная сессия L3000 оборвалась после первых 10 структур, затем штатный CLI продолжил недостающие входы с сохранением готовых файлов. Возобновление перезапускает RNG с seed=0; последовательность случайных выборок отличается от непрерывного запуска, что учтено как ограничение, а не скрыто.

Автор issue не опубликовал исходные 140 прогнозов, точный seed, MSA и YAML. Точные CASP16 входы и оценочный код статьи также не представлены в закреплённом upstream. Поэтому наш результат является независимой реконструкцией с известными входами и настройками; точное происхождение каждой сотой в чужом результате не доказано. Один seed не оценивает вариацию между независимыми запусками. Никаких молекул, seed, голов ансамбля или коэффициентов не выбирали по достижению 0.65.

## Как повторить

```bash
/mnt/data/arman-projects/boltz-h200/.venv/bin/python /mnt/data/arman-projects/boltz-h200/research/casp16/prepare.py prepare
/mnt/data/arman-projects/boltz-h200/.venv/bin/python /mnt/data/arman-projects/boltz-h200/research/casp16/prepare.py msa --target L1000
/mnt/data/arman-projects/boltz-h200/.venv/bin/python /mnt/data/arman-projects/boltz-h200/research/casp16/run_study.py
/mnt/data/arman-projects/boltz-h200/.venv/bin/python /mnt/data/arman-projects/boltz-h200/research/casp16/score.py /mnt/data/arman-projects/boltz-h200/research/casp16/runs/issue-settings-csv-seed0 --require-complete
```

`score.py` соединяет прогнозы и метки по Target ID, проверяет полное покрытие и вычисляет оба способа усреднения. Для новых CLI-прогнозов с официальной поправкой можно указать `--affinity_mw_correction`; к уже исправленному выходу не нужно применять её повторно.

## Первичные источники

- [Исходный issue #636](https://github.com/jwohlwend/boltz/issues/636)
- [Статья Boltz-2: §5, B.5.2, D.2.2, Table 14](https://jeremywohlwend.com/assets/boltz2.pdf)
- [Код ансамбля и MW-поправки](https://github.com/jwohlwend/boltz/blob/b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc/src/boltz/model/models/boltz2.py#L659)
- [CLI с default False для MW-поправки](https://github.com/jwohlwend/boltz/blob/b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc/src/boltz/main.py#L992)
- [Официальные CASP16 SMILES и affinity CSV](https://predictioncenter.org/download_area/CASP16/targets/pharma_ligands/)
'''
    (destination / "REPORT_RU.md").write_text(report)
    print(json.dumps({"report": str(destination / "REPORT_RU.md"), "raw_uniform": raw_uniform,
                      "raw_weighted": raw_weighted, "mw_uniform": corrected_uniform,
                      "mw_weighted": corrected_weighted, "coverage": summary["coverage"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    main(args.run_root.resolve())
