import os
import time
import argparse
import numpy as np
import lightgbm as lgb
from ember import read_vectorized_features
from ember.features import PEFeatureExtractor


def list_all_files(data_dir):
    all_files = []
    mal_path = os.path.join(data_dir, 'MALICIOUS')
    mal_files = [(1, os.path.join(mal_path, fp)) for fp in os.listdir(mal_path)]
    all_files.extend(mal_files)
    clean_path = os.path.join(data_dir, 'KNOWN')
    clean_files = [(0, os.path.join(clean_path, fp)) for fp in os.listdir(clean_path)]
    all_files.extend(clean_files)
    return all_files


def benchmark_dataset(lgbm_model, data_dir, feature_version):
    all_infer_time = []
    all_files = list_all_files(data_dir)
    extractor = PEFeatureExtractor(feature_version)

    from progress.bar import Bar
    bar = Bar('Progress... ', max=len(all_files))
    for label, binary_path in all_files:
        if not os.path.exists(binary_path):
            print("{} does not exist".format(binary_path))

        try:
            file_data = open(binary_path, "rb").read()
            features = np.array(extractor.feature_vector(file_data), dtype=np.float32)
            start = time.time()
            score = lgbm_model.predict([features])[0]
            finish = time.time()
            all_infer_time.append(1000 * (finish - start))
            bar.next()
        except Exception:
            bar.next()
            continue
    bar.finish()

    print(f'average inference time: {np.mean(all_infer_time)} ms')


def benchmark_jsonl(lgbm_model, X_test):
    start = time.time()
    prediction = lgbm_model.predict(X_test)
    finish = time.time()
    avg_infer_time = 1000 * (finish - start) / len(X_test)
    return prediction, avg_infer_time


def benchmark_jsonl_daal(lgbm_model, X_test):
    import daal4py as d4p
    daal_model = d4p.get_gbt_model_from_lightgbm(lgbm_model)
    predict_algo = d4p.gbt_classification_prediction(nClasses=2)

    start = time.time()
    daal_prediction = predict_algo.compute(X_test, daal_model).prediction
    finish = time.time()

    avg_infer_time = 1000 * (finish - start) / len(X_test)
    return daal_prediction, avg_infer_time


def main():
    prog = "classify_binaries"
    descr = "Use a trained ember model to make predictions on PE files"
    parser = argparse.ArgumentParser(prog=prog, description=descr)
    parser.add_argument("-v", "--featureversion", type=int, default=2, help="EMBER feature version")
    parser.add_argument("-m", "--modelpath", type=str, default=None, required=True, help="Ember model")
    parser.add_argument("binaries", metavar="BINARIES", type=str, nargs="+", help="PE files to classify")
    args = parser.parse_args()

    if not os.path.exists(args.modelpath):
        parser.error("ember model {} does not exist".format(args.modelpath))
    lgbm_model = lgb.Booster(model_file=args.modelpath)

    data_dir = args.binaries[0]
    if os.path.exists(os.path.join(data_dir, 'MALICIOUS')):
        benchmark_dataset(lgbm_model, data_dir, args.featureversion)
    elif os.path.exists(os.path.join(data_dir, 'X_test.dat')):
        X_test, y_test = read_vectorized_features(data_dir, "test", args.featureversion)
        predict0, t0 = benchmark_jsonl(lgbm_model, X_test)
        predict1, t1 = benchmark_jsonl_daal(lgbm_model, X_test)
        # np.save('ember.gbt.predict.npy', predict0)
        # np.save('ember.daal4py.predict.npy', predict1)

        print(f'daal boost: {t0}/{t1} = {t0 / t1:.2f}')

        totals = len(predict0)
        abs_diff = np.absolute(predict0.flatten() - predict1.flatten())
        for x in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
            count = np.sum(abs_diff < x)
            print(f'abs_diff<{x:.1e}\t{count}/{totals}={count / totals:.0%}')


if __name__ == "__main__":
    main()
