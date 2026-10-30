"""核查论文中核心30项众数混淆矩阵的计数。"""
import csv
from collections import defaultdict
from pathlib import Path

path = Path(__file__).resolve().parents[1] / 'outputs/submission_review_20261029/ordinal_confusion.csv'
records = defaultdict(list)
with path.open(encoding='utf-8-sig', newline='') as stream:
    for row in csv.DictReader(stream):
        if row['scope'] == 'core30':
            records[row['model']].append(tuple(int(row[x]) for x in ('observed_grade', 'predicted_grade', 'count')))

def summarize(rows):
    return {
        'total': sum(n for a, b, n in rows),
        'correct': sum(n for a, b, n in rows if a == b),
        'correct_zero': sum(n for a, b, n in rows if a == b == 0),
        'correct_nonzero': sum(n for a, b, n in rows if a == b and a > 0),
        'nonzero_to_zero': sum(n for a, b, n in rows if a > 0 and b == 0),
        'nonzero_to_wrong_nonzero': sum(n for a, b, n in rows if a > 0 and b > 0 and a != b),
    }

for model in ('population', 'style_only', 'frequency_ET'):
    result = summarize(records[model])
    assert result['total'] == 7500, (model, result)
    print(model, result)
assert summarize(records['frequency_ET']) == dict(total=7500, correct=6039, correct_zero=5638, correct_nonzero=401, nonzero_to_zero=459, nonzero_to_wrong_nonzero=788)
