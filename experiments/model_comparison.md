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
