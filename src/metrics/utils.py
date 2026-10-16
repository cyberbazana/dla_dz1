# Based on seminar materials

# Don't forget to support cases when target_text == ''

def calc_dist(target_text, predicted_text):
    if len(target_text) < len(predicted_text):
        target_text, predicted_text = predicted_text, target_text
    d_prev = list(range(len(predicted_text) + 1))
    for i, target_item in enumerate(target_text, start=1):
        d_curr = [i]
        for j, pred_item in enumerate(predicted_text, start=1):
            cost = (target_item != pred_item)
            d_curr.append(min(d_prev[j] + 1, d_curr[j - 1] + 1, d_prev[j - 1] + cost))
        d_prev = d_curr
    return d_prev[-1]


def calc_cer(target_text, predicted_text) -> float:
    target_text = list(target_text)
    predicted_text = list(predicted_text)
    return calc_dist(target_text, predicted_text) / max(len(target_text), 1)


def calc_wer(target_text, predicted_text) -> float:
    target_text = target_text.split()
    predicted_text = predicted_text.split()
    return calc_dist(target_text, predicted_text) / max(len(target_text), 1)
