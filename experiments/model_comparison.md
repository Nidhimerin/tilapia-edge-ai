# Fish Detection Model Comparison

## Objective
Compare the original YOLO model with the fine-tuned model using validation images from the combined tilapia dataset.

## Dataset
- Training: 23 images
- Validation: 9 images
- Test: 2 images
- Classes: 1 (`fish`)

## Validation Results

| Metric | Original Model | Fine-tuned Model |
|---|---:|---:|
| Precision | 0.8571 | 0.8077 |
| Recall | 0.2857 | 0.3333 |
| mAP@50 | 0.2791 | 0.3232 |
| mAP@50–95 | 0.2100 | 0.2450 |

## Interpretation
The fine-tuned model achieved higher mAP@50 and mAP@50–95, and recall improved slightly. Precision decreased compared with the original model.

These results are preliminary because validation used only nine images. The dataset should be expanded and the annotations reviewed before drawing firm conclusions.

## Counting Evaluation
A separate counting evaluation on nine validation images reported:
- Original model MAE: 2.67 fish
- Fine-tuned model MAE: 3.67 fish

The fine-tuned model's detection metrics improved, but its counting MAE was worse in that evaluation. Therefore, improved detection metrics do not yet establish improved fish-counting accuracy.

## Next Steps
1. Review missed and duplicate fish annotations.
2. Expand the validation and test datasets.
3. Evaluate both models on the same held-out images and confidence threshold.
4. Reassess counting MAE after annotation verification.

**Note:** These results are preliminary and do not establish final deployment performance.

### Verified Fish-Counting Evaluation

**Dataset:** 9 validation images  
**Ground-truth count:** 63 fish instances  
**Metric:** Mean Absolute Error (MAE), measured in fish per image.

| Model | Best tested confidence | MAE |
|---|---:|---:|
| Original | 0.10 | 2.67 |
| Fine-tuned | 0.15 | 3.22 |

**Observations**
- The original model achieved the lowest observed counting MAE at a confidence threshold of 0.10.
- The fine-tuned model achieved a lower MAE than the original at confidence thresholds 0.15, 0.20, 0.30, and 0.40.
- At confidence 0.25, both models had the same MAE.
- These results are preliminary because the evaluation uses only nine validation images.

**Conclusion:** Fine-tuning has not yet demonstrated a lower best-observed counting MAE than the original model. Additional evaluation on independent tank images is needed before drawing broader conclusions.

**Reproducibility note:** Confirm that the predictions and MAE calculations were generated using the manually verified ground-truth counts before treating these figures as final.
