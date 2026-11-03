"""Summarize completed affinity alternatives; do not select inference inputs."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPERIMENTS = HERE/'experiments'
CONTEXT = EXPERIMENTS/'context300-seed1-20261103'


def main():
    context = json.loads((CONTEXT/'analysis.json').read_text())
    previous = json.loads((EXPERIMENTS/'POSE_EXPERIMENTS_SUMMARY.json').read_text())
    verified = json.loads((CONTEXT/'independent-verification.json').read_text())
    integrity = json.loads((CONTEXT/'final-integrity.json').read_text())
    assert verified['passed'] and integrity['passed'] and context['pairs'] == 140
    rows = list(previous['rows'])
    for mode, label in [('context300','Контекст белка до 300 токенов, seed1'),
                        ('matched_ensemble','Среднее контекстов 200 и 300, seed1'),
                        ('original_ensemble','Среднее исторического расчёта и контекста300')]:
        rows.append({'method':label,
            'r_cli_mw':context['aggregate'][mode+'_processed']['weighted'],
            'r_csv_mw':context['aggregate'][mode+'_csv']['weighted']})
    best = max(rows,key=lambda row:row['r_cli_mw'])
    result = {'pairs':140,'targets':2,'rows':rows,'highest_observed':best,
        'context300_vs_original_delta':context['aggregate']['context300_processed']['weighted']-context['aggregate']['original_processed']['weighted'],
        'context300_vs_control_delta':context['aggregate']['context300_processed']['weighted']-context['aggregate']['control_processed']['weighted'],
        'bootstrap':context['bootstrap'],'reached_0_8':best['r_cli_mw']>=.8,
        'caveat':'Exploratory repeated benchmark use; CIs unadjusted for multiple comparisons. Same CLI seed/order do not fix noise when tensor shapes change. No independent protein validation.'}
    (EXPERIMENTS/'AFFINITY_EXPERIMENTS_SUMMARY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines = ['# Affinity: итог расширения контекста и прежних экспериментов','',
        f"На тех же 140 парах наибольшее наблюдённое взвешенное R теперь {best['r_cli_mw']:.6f}; исторический результат 0.641163. Цель 0.8 пока не достигнута.",'',
        'В новом прогоне увеличен только контекст affinity: до 300 белковых токенов вместо 200, до 384 токенов всего и 3072 атомов. У L1000 в контекст попадает весь белок (247 аминокислот), у L3000 — 290–300. Во всех случаях сохранены весь лиганд и исходная область контекста. Веса модели, химия, структура и параметры диффузии не менялись.','',
        '| Метод | Взвешенный R, MW CLI | R, MW CSV |','|---|---:|---:|']
    for row in rows:
        lines.append(f"| {row['method']} | {row['r_cli_mw']:.6f} | {row['r_csv_mw']:.6f} |")
    lines += ['','## Результат по белкам','',
        '| Белок | Пар | Исторический расчёт | Контекст200, seed1 | Контекст300, seed1 |',
        '|---|---:|---:|---:|---:|']
    for target, group in context['per_target'].items():
        r=group['r']
        lines.append(f"| {target} | {group['n']} | {r['original_processed']:.6f} | {r['control_processed']:.6f} | {r['context300_processed']:.6f} |")
    lines += ['', 'Основной прирост пришёлся на L3000; у L1000 новый результат ниже исторического. Взвешенная метрика усредняет две корреляции с весами 17/140 и 123/140, а не объединяет все молекулы в одну корреляцию.','',
        f"Изменение нового контекста относительно исторического: {result['context300_vs_original_delta']:+.6f}; относительно согласованного seed1-контроля: {result['context300_vs_control_delta']:+.6f}.",'',
        '## Насколько уверен результат','',
        '95% парный стратифицированный bootstrap-интервал изменения контекст300 относительно исторического расчёта: '+str(context['bootstrap']['context300_vs_original_delta_ci95'])+'. Интервал включает ноль: наблюдаемое повышение ещё не доказывает устойчивого улучшения. Относительно seed1-контроля интервал также включает ноль.','',
        'У заранее заданного среднего исторического и нового прогнозов R=0.666584; интервал изменения '+str(context['bootstrap']['original_ensemble_vs_original_delta_ci95'])+'. Это положительный исследовательский сигнал. Интервалы не скорректированы за множество уже проверенных вариантов, поэтому независимые белки и повторные seed остаются необходимы.','',
        'Исходные pre_affinity структуры побитно совпадают с контролем. Одинаковый seed и порядок входов не гарантируют одинаковый шум после изменения размеров тензоров. Это изменённая конфигурация инференса, не повтор штатных настроек статьи. Модель не дообучалась и метки не использовались для выбора контекста или весов ансамбля.','',
        '## Проверка и ресурсы','',
        '- Все 140 новых прогнозов конечны, ID уникальны и полны; среднее двух голов проверено.',
        '- Независимый Pearson через стандартную библиотеку совпал со SciPy до 1e-12.',
        '- Сохранность исходных прогнозов, входных структур, меток, весов и upstream-кода проверена.',
        '- Четыре CLI-процесса завершились с кодом 0. Только стадия affinity на сохранённых структурах; 2148.98 с (35.82 мин), до двух процессов одновременно на H200 №2.',
        '- Наблюдавшиеся пики отдельных процессов NVML: 17 392, 28 412, 21 412, 31 508 MiB. Опрос 0.5 с; это не точные пики аллокатора и не сопоставление E2E-скорости с исходным прогоном.',
        '- Все наши GPU-процессы завершены. Это исследовательская проверка, не официальный upstream test suite.','',
        'В коде выходная голова усредняет взаимодействия лиганд–белок и лиганд–лиганд. Расширение контекста меняет и эту область усреднения. Это механическое наблюдение, а не доказанная причина изменения качества.','',
        'Основной столбец использует точный признак MW из processed manifest CLI. Второй сохраняет прежнее определение MolWt по CSV SMILES. Метод поправки и метрика едины внутри каждого столбца.','',
        'Подробности нового прогона: [отчёт](context300-seed1-20261103/REPORT_RU.md), [все 140 прогнозов](context300-seed1-20261103/all-predictions.csv), [независимая проверка](context300-seed1-20261103/independent-verification.json).']
    (EXPERIMENTS/'AFFINITY_EXPERIMENTS_SUMMARY_RU.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'highest_observed':best,'context300_vs_original_delta':result['context300_vs_original_delta'],'summary':str(EXPERIMENTS/'AFFINITY_EXPERIMENTS_SUMMARY_RU.md')},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
