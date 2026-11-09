import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'reproduce'))
import aggregate  # noqa: E402

CELLS = """dataset,arm,seed,policy,T,metric,pxacc
ade20k,learned,1,learned,4,24.0,70.0
ade20k,learned,2,learned,4,24.2,70.2
ade20k,learned,3,learned,4,24.4,70.4
ade20k,random,1,imagenet_policy,4,23.0,68.0
imagenet,learned,1,learned,4,68.0,
"""


def _cells(tmp_path):
    p = tmp_path / 'cells.csv'
    p.write_text(CELLS)
    return aggregate.load(p)


def test_mean_and_sample_sd(tmp_path):
    assert aggregate.fmt(_cells(tmp_path)['ade20k', 'learned', 'learned', 4]['metric']) == '24.2 ± 0.2'


def test_counts_and_missing():
    assert aggregate.fmt([23.0]) == '23.0 (n=1)'
    assert aggregate.fmt([]) == '--'


def test_table3_native_cell_is_the_table2_cell(tmp_path):
    cells = _cells(tmp_path)
    # learned ADE20K model with the ADE20K policy = its own learned policy
    r = aggregate.row(cells, 'learned', 'ade20k_policy', 4, aggregate.COLUMNS[2])
    assert r[1:3] == ['24.2 ± 0.2', '70.2 ± 0.2']
    # a random-location model has no native learned policy, the transplanted cell is read as is
    r = aggregate.row(cells, 'random', 'imagenet_policy', 4, aggregate.COLUMNS[2])
    assert r[1] == '23.0 (n=1)'


def test_check_flags_only_cells_outside_tolerance(tmp_path):
    ours, exp = tmp_path / 'cells.csv', tmp_path / 'expected.csv'
    head = 'dataset,arm,seed,policy,T,metric,pxacc\n'
    exp.write_text(head + 'ade20k,learned,1,learned,4,24.00,70.0\nade20k,random,1,random,4,21.0,68.0\n'
                   'imagenet,learned,1,learned,4,68.0,\n')
    ours.write_text(head + 'ade20k,learned,1,learned,4,24.20,70.0\n'   # deterministic, off by 0.2 -> flagged
                    'ade20k,random,1,random,4,21.3,68.2\n'             # sampled, within 0.5 -> ok
                    'imagenet,learned,1,learned,4,68.0,\n')
    report = aggregate.check(ours, exp)
    assert report[0].endswith('5 values compared, 1 outside tolerance')
    assert len(report) == 3 and 'ade20k,learned,1,learned,4 metric' in report[2]


def test_check_counts_missing_values_per_dataset(tmp_path):
    ours, exp = tmp_path / 'cells.csv', tmp_path / 'expected.csv'
    head = 'dataset,arm,seed,policy,T,metric,pxacc\n'
    exp.write_text(head + 'ade20k,learned,1,learned,4,24.0,70.0\nade20k,random,1,random,4,21.0,68.0\n'
                   'imagenet,learned,1,learned,4,68.0,\n')
    ours.write_text(head + 'ade20k,learned,1,learned,4,24.0,70.0\n')   # one ade20k cell, no imagenet cell
    assert aggregate.check(ours, exp)[1] == 'Missing from cells.csv: ade20k 2 of 4, imagenet 1 of 1 (not run)'
