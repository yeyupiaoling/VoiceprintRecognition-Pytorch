from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate

metric = DiarizationErrorRate()


false_alarms, confusions, missed_detections, error_rates = [], [], [], []
references = load_rttm('dataset/references.rttm')
hypotheses = load_rttm('dataset/hypotheses.rttm')
for uri, reference in references.items():
    hypothesis = hypotheses[uri]
    result = metric(reference, hypothesis, detailed=True)
    print(uri, ":", result)
    false_alarms.append(result["false alarm"])
    confusions.append(result["confusion"])
    missed_detections.append(result["missed detection"])
    error_rates.append(result["diarization error rate"])
print("False alarm:", round(sum(false_alarms) / len(false_alarms), 5))
print("Confusion:", round(sum(confusions) / len(confusions), 5))
print("Missed detection:", round(sum(missed_detections) / len(missed_detections), 5))
print("Diarization error rate:", round(sum(error_rates) / len(error_rates), 5))
