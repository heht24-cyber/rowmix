# Data and validation sources

The Digits example uses the 1,797-image scikit-learn distribution of Optical Recognition of Handwritten Digits by E. Alpaydin and C. Kaynak. The Wine example uses the UCI Wine dataset. The calibration arrays, eight-example inputs and labels are subsets converted to NumPy arrays or little-endian float records. Models were trained for this project and the published split IDs are identified by hashes in benchmarks/results.json.

- [Optical Recognition of Handwritten Digits, UCI, DOI 10.24432/C50P49](https://doi.org/10.24432/C50P49)
- [Wine, UCI, DOI 10.24432/C5PC7J](https://doi.org/10.24432/C5PC7J)
- Both source datasets are distributed under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).

The interchange suite reads named test fixtures from the separately installed ONNX 1.18.0 distribution. That project provides its own license and attributions. No upstream fixture weights or ONNX Runtime binaries are redistributed in this source package.
