"""Fixed-threshold instance metrics; not COCO mAP and not a tracking benchmark."""
from collections import Counter
import numpy as np
from robot_skill_stack.world.perception.v2.types import CLASSES


def iou_matrix(truth, predictions):
    out = np.zeros((len(truth), len(predictions)))
    for i, gt in enumerate(truth):
        for j, pred in enumerate(predictions):
            a, b = gt['mask'], pred['mask']
            if a.shape != b.shape:
                raise ValueError('Mask shapes differ')
            out[i, j] = np.count_nonzero(a & b) / max(1, np.count_nonzero(a | b))
    return out


def assign(iou, allowed):
    """Maximum cardinality, then maximum total IoU; one-to-one Hungarian assignment."""
    rows, cols = iou.shape
    n = max(rows, cols)
    if not n:
        return []
    cost = np.zeros((n, n))
    cost[:rows, :cols] = -np.where(allowed, 1+iou/(n+1), 0.)
    u, v, p, way = np.zeros(n+1), np.zeros(n+1), np.zeros(n+1, int), np.zeros(n+1, int)
    for i in range(1, n+1):
        p[0], j0 = i, 0
        minv, used = np.full(n+1, np.inf), np.zeros(n+1, bool)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], np.inf, 0
            for j in range(1, n+1):
                if used[j]:
                    continue
                cur = cost[i0-1, j-1]-u[i0]-v[j]
                if cur < minv[j]:
                    minv[j], way[j] = cur, j0
                if minv[j] < delta:
                    delta, j1 = minv[j], j
            for j in range(n+1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    return [(p[j]-1, j-1) for j in range(1, n+1)
            if 0 < p[j] <= rows and j <= cols and allowed[p[j]-1, j-1]]


def ratio(n, d):
    return None if not d else float(n/d)


def summarize(values):
    return dict(n=len(values), mean=None if not values else float(np.mean(values)),
                median=None if not values else float(np.median(values)),
                p90=None if not values else float(np.percentile(values, 90)))


class Metrics:
    def __init__(self, threshold=.5):
        if not 0 < threshold <= 1:
            raise ValueError('IoU threshold must be in (0,1]')
        self.threshold = threshold
        self.counts, self.gt, self.pred, self.tp = Counter(), Counter(), Counter(), Counter()
        self.confusion = {c: {p: 0 for p in (*CLASSES, 'unknown', 'missed')} for c in (*CLASSES, 'background')}
        self.ious, self.center, self.size, self.latency = [], [], [], []
        self.elevated_center, self.supported_center = [], []

    def add(self, truth, predictions, seconds):
        matrix = iou_matrix(truth, predictions)
        allowed = matrix >= self.threshold
        pairs = assign(matrix, allowed)
        same = np.array([[a['label'] == b['label'] for b in predictions] for a in truth], bool).reshape(matrix.shape)
        correct_pairs = assign(matrix, allowed & same)
        self.counts.update(frames=1, gt=len(truth), predictions=len(predictions), localized=len(pairs),
                           correct_localized=sum(truth[i]['label'] == predictions[j]['label'] for i, j in pairs),
                           unknown=sum(p['label'] is None for p in predictions),
                           negatives=int(not truth), negative_false_positive_frames=int(not truth and bool(predictions)),
                           fitted=sum(p.get('geometry_known', False) for p in predictions))
        self.gt.update(t['label'] for t in truth)
        self.pred.update(p['label'] for p in predictions)
        self.tp.update(truth[i]['label'] for i, _ in correct_pairs)
        self.latency.append(seconds*1000)
        matched_gt, matched_pred = {i for i, _ in pairs}, {j for _, j in pairs}
        for i, j in pairs:
            a, b = truth[i], predictions[j]
            self.confusion[a['label']][b['label'] or 'unknown'] += 1
            self.ious.append(float(matrix[i, j]))
            if b.get('position') is not None:
                error = 1000*float(np.linalg.norm(a['position']-b['position']))
                self.center.append(error)
                (self.elevated_center if a.get('elevated') else self.supported_center).append(error)
                self.size.append(1000*float(np.linalg.norm(a['size']-b['size'])))
        for i, t in enumerate(truth):
            if i not in matched_gt:
                self.confusion[t['label']]['missed'] += 1
        for j, p in enumerate(predictions):
            if j not in matched_pred:
                self.confusion['background'][p['label'] or 'unknown'] += 1
        return pairs

    def report(self):
        classes = {}
        for c in CLASSES:
            tp, ng, npred = self.tp[c], self.gt[c], self.pred[c]
            classes[c] = dict(gt=ng, predicted=npred, tp=tp, fp=npred-tp, fn=ng-tp,
                              precision=ratio(tp, npred), recall=ratio(tp, ng), f1=ratio(2*tp, ng+npred))
        correct = sum(self.tp.values())
        known = sum(self.pred[c] for c in CLASSES)
        ng = self.counts['gt']
        return dict(counts=dict(self.counts), per_class=classes,
                    named_target_precision=ratio(correct, known), named_target_recall=ratio(correct, ng),
                    named_target_f1=ratio(2*correct, known+ng),
                    localization_precision=ratio(self.counts['localized'], self.counts['predictions']),
                    localization_recall=ratio(self.counts['localized'], ng),
                    classification_accuracy_on_localized=ratio(self.counts['correct_localized'], self.counts['localized']),
                    mean_matched_mask_iou=ratio(sum(self.ious), len(self.ious)), confusion=self.confusion,
                    center_error_mm=summarize(self.center), size_vector_error_mm=summarize(self.size),
                    elevated_center_error_mm=summarize(self.elevated_center),
                    supported_center_error_mm=summarize(self.supported_center), latency_ms=summarize(self.latency))
