"""Collect all completed alternatives without hiding negative comparisons."""

import json
from pathlib import Path


HERE=Path(__file__).resolve().parent
EXPERIMENTS=HERE/'experiments'


def main():
    pose_root=EXPERIMENTS/'pose-rank1-seed1-20261103'
    select_root=EXPERIMENTS/'ligand-iptm-selector-seed1-20261103'
    pose=json.loads((pose_root/'analysis.json').read_text())
    selector=json.loads((select_root/'analysis.json').read_text())
    assert json.loads((pose_root/'independent-verification.json').read_text())['passed']
    assert json.loads((select_root/'independent-verification.json').read_text())['passed']
    rows=[]
    for key,label in [('original','Исходный исторический расчёт'),('control','Предыдущий контроль: первая поза, seed1'),
                      ('pose1','Вторая поза по общей уверенности, seed1'),('matched_ensemble','Среднее первой и второй позы, seed1'),
                      ('original_pose1_ensemble','Среднее исторического расчёта и второй позы')]:
        rows.append({'method':label,'r_cli_mw':pose['aggregate'][key+'_processed']['weighted'],
                     'r_csv_mw':pose['aggregate'][key+'_csv']['weighted']})
    for key,label in [('selected','Поза по максимальному ligand_iptm'),('ensemble','Среднее контроля и выбора по ligand_iptm')]:
        rows.append({'method':label,'r_cli_mw':selector['weighted_r'][key+'_processed'],
                     'r_csv_mw':selector['weighted_r'][key+'_csv']})
    best=max(rows,key=lambda r:r['r_cli_mw'])
    result={'pairs':140,'rows':rows,'highest_observed':best,'reached_0_8':best['r_cli_mw']>=.8,
            'new_cli_processes_this_round':6,'new_affinity_calculations_this_round':163,
            'note':'All alternatives and negative results retained. The largest observed score is exploratory, not an independent generalization estimate.'}
    dest=EXPERIMENTS/'POSE_EXPERIMENTS_SUMMARY_RU.md'
    lines=['# Альтернативные позы: результаты двух экспериментов','',
           'На всех 140 парах проверены вторая поза и выбор позы по ligand_iptm. Правила выбора и веса средних зафиксированы до соответствующих affinity-вычислений. Модель не дообучалась; коэффициенты MW-поправки не подбирались.','',
           '| Вариант | Взвешенный R, MW CLI | R, MW исходного CSV |','|---|---:|---:|']
    for row in rows:lines.append(f"| {row['method']} | {row['r_cli_mw']:.6f} | {row['r_csv_mw']:.6f} |")
    lines+=['',f"Наибольшее наблюдённое R: {best['r_cli_mw']:.6f} ({best['method']}). Это выбор максимума среди исследовательских сравнений, не доказательство улучшения на новых белках.",'',
            'В этом раунде запущены шесть CLI-процессов и получены 163 новых affinity-прогноза: 140 для второй позы и 23 для оставшихся выбранных поз. Остальные 117 значений второго метода переиспользованы из контролей первой/второй позы. 163 означает повторные расчёты на прежних молекулах, а не 163 новых уникальных пары.','',
            '## Что именно проверено','',
            '- Первая серия: confidence rank=1 вместо rank=0, тот же seed1 и порядок входов. Одна жёсткая пространственная трансформация совмещает карман с контролем, сохраняя внутренние расстояния. Перенос атомов по именам/остаткам проверен. Изменение crop может менять размеры тензоров и потребление RNG, поэтому одинаковый seed не означает побитно одинакового шума.',
            '- Вторая серия: максимальный ligand_iptm среди пяти исходных поз, без использования экспериментальных affinity. У 61 из 140 молекул выбор отличается от стандартного rank0; 23 нужных прогноза ещё не были посчитаны. Последовательности RNG свежих малых CLI отличаются от полного контроля; это отмечено как ограничение.',
            '- Штатный affinity crop ограничен 200 белковыми токенами, 256 токенами всего и 2048 атомами. У L3000 альтернативная поза меняет выбранную область во всех 123 случаях; медианный Jaccard белковых областей около 0.888. У L1000 области гораздо ближе.',
            '- Все 140 ID сохранены, пропусков нет. Независимый расчёт корреляций совпал с основным до 1e-12. Пары исходных файлов не менялись. Это исследовательские проверки полноты и арифметики, не официальный upstream test suite.','',
            'Ранее опубликованные 0.641022 используют MolWt исходных CSV SMILES. Здесь основной столбец воспроизводит именно сохранённый признак массы CLI и исходно даёт 0.641163. Оба определения массы показаны; они не смешаны в одном сравнении.','',
            'Время повторного affinity на подготовленных структурах не является E2E benchmark. Подробные интервалы и ресурсы лежат в отдельных каталогах:','',
            '- [Вторая поза](pose-rank1-seed1-20261103/REPORT_RU.md)',
            '- [Выбор по ligand_iptm](ligand-iptm-selector-seed1-20261103/REPORT_RU.md)']
    dest.write_text('\n'.join(lines)+'\n')
    (EXPERIMENTS/'POSE_EXPERIMENTS_SUMMARY.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
