# Prediction Results Visualizer

This directory contains a visual report generator for the C++ batch inference outputs under
`runs/dgx_spark_full_cpp/cpp_full_test`.

## What It Reads

- `runs/dgx_spark_full_cpp/cpp_full_test/predictions.tsv`
- `runs/dgx_spark_full_cpp/cpp_full_test/cpp_infer.log`
- `runs/dgx_spark_full_cpp/*.pth`
- `runs/dgx_spark_full_cpp/*.pt`

## What It Generates

- `prediction_results/cpp_full_test_report.html`

The HTML report includes:

- overall accuracy and confidence summary
- confusion matrix
- class-by-class recall table
- image galleries for wrong predictions and low-confidence predictions
- an explanation of what the generated `.pth` and `.pt` files mean

## Run

From the project root:

```bash
python3 prediction_results/visualize_cpp_results.py
```

If you want a different output file:

```bash
python3 prediction_results/visualize_cpp_results.py \
  --output prediction_results/my_report.html
```

## File Meaning

- `.pth`: Python-side PyTorch checkpoints. They store `model_state_dict` and training metadata.
- `.pt`: the TorchScript export used by the LibTorch C++ inference binary.
# cnn_transformer_hybrid
