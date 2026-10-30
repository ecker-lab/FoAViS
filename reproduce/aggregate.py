"""Print Tables 2 and 3 of the paper from the cells written by table2.sh / table3.sh, and check every cell
against the values of the released checkpoints (reproduce/expected.csv).

Each entry is mean +- sample standard deviation over the seeds present; entries with fewer than three seeds
show their count, missing entries show --.

    python reproduce/aggregate.py [reproduce_out/cells.csv]
"""
import csv
import os
import statistics
import sys
from collections import defaultdict

EXPECTED = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'expected.csv')
# allowed |difference| to expected.csv: random-location cells draw new fixations at every evaluation,
# all other policies are deterministic and differ only by GPU non-determinism
TOLERANCE = {'sampled': 0.5, 'deterministic': 0.05}

# (dataset, metric column) per table column
COLUMNS = {
    1: [('cmnist', 'metric'), ('imagenet', 'metric'), ('ade20k', 'metric'), ('ade20k', 'pxacc'),
        ('coco_stuff', 'metric'), ('coco_stuff', 'pxacc')],
    2: [('imagenet', 'metric'), ('ade20k', 'metric'), ('ade20k', 'pxacc'),
        ('coco_stuff', 'metric'), ('coco_stuff', 'pxacc')],
}
HEADER = {'metric': {'cmnist': 'C-MNIST Acc', 'imagenet': 'ImageNet Acc', 'ade20k': 'ADE20K mIoU',
                     'coco_stuff': 'COCO-Stuff mIoU'},
          'pxacc': {'ade20k': 'ADE20K Px Acc', 'coco_stuff': 'COCO-Stuff Px Acc'}}
ARMS = {'random': 'Trained with random locations', 'learned': 'Trained with its own learned policy'}
# Table 3 rows: (label, policy name in cells.csv, glimpses)
TABLE3_ROWS = [('random locations', 'random', 4), ('triangle', 'triangle', 4),
               ('quadrant', 'quadrant', 5), ('saliency + IoR', 'saliency_ior', 4),
               ('entropy + IoR', 'entropy_ior', 4), ('ImageNet policy', 'imagenet_policy', 4),
               ('ADE20K policy', 'ade20k_policy', 4), ('COCO-Stuff policy', 'coco_stuff_policy', 4)]


def load(path):
    """cells.csv -> {(dataset, arm, policy, T): {'metric': [...], 'pxacc': [...]}}, one value per seed."""
    cells = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for r in csv.DictReader(f):
            for k in ('metric', 'pxacc'):
                if r[k]:
                    cells[r['dataset'], r['arm'], r['policy'], int(r['T'])][k].append(float(r[k]))
    return cells


def load_rows(path):
    """cells.csv -> {(dataset, arm, seed, policy, T): {'metric': float, 'pxacc': float}} (empty fields dropped)."""
    with open(path) as f:
        return {(r['dataset'], r['arm'], r['seed'], r['policy'], r['T']):
                {k: float(r[k]) for k in ('metric', 'pxacc') if r[k]} for r in csv.DictReader(f)}


def check(path, expected_path=EXPECTED):
    """Compare every evaluated cell with expected.csv; returns the report lines."""
    ours, exp = load_rows(path), load_rows(expected_path)
    bad, n = [], 0
    for key, vals in sorted(ours.items()):
        if key not in exp:
            continue
        kind = 'sampled' if key[3] == 'random' else 'deterministic'
        for k, v in vals.items():
            if k in exp[key]:
                n += 1
                d = v - exp[key][k]
                if abs(d) > TOLERANCE[kind]:
                    bad.append(f'  {",".join(key)} {k}: {v:.3f} vs expected {exp[key][k]:.3f} ({d:+.3f}, {kind})')
    head = f'Check against {os.path.basename(expected_path)}: {n} values compared, {len(bad)} outside tolerance'
    return [head] + bad


def fmt(values):
    if not values:
        return '--'
    m = statistics.mean(values)
    s = f'{m:.1f}' if len(values) == 1 else f'{m:.1f} ± {statistics.stdev(values):.1f}'
    return s if len(values) >= 3 else f'{s} (n={len(values)})'


def row(cells, arm, policy, T, columns):
    # in Table 3 the learned model's own policy is the native learned cell
    return [fmt(cells.get((ds, arm, 'learned' if arm == 'learned' and policy == f'{ds}_policy' else policy, T),
                          {}).get(k, [])) for ds, k in columns]


def table(rows, columns, first):
    head = first + [HEADER[k][ds] for ds, k in columns]
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for r in rows:
        lines.append('| ' + ' | '.join(r) + ' |')
    return '\n'.join(lines)


def table2(cells):
    rows = [[ARMS[arm], str(T)] + row(cells, arm, arm, T, COLUMNS[1])
            for arm in ARMS for T in (1, 4, 6)]
    return table(rows, COLUMNS[1], ['Model', '# Glimpses'])


def table3(cells):
    rows = [[ARMS[arm], label, str(T)] + row(cells, arm, policy, T, COLUMNS[2])
            for arm in ARMS for label, policy, T in TABLE3_ROWS]
    return table(rows, COLUMNS[2], ['Model', 'Evaluation policy', '# Glimpses'])


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'reproduce_out/cells.csv'
    cells = load(path)
    print('Table 2: native policy evaluation\n\n' + table2(cells) + '\n')
    print('Table 3: policy transfer evaluation\n\n' + table3(cells) + '\n')
    if os.path.exists(EXPECTED):
        print('\n'.join(check(path)))
