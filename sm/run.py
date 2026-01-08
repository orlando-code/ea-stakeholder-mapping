"""Run the entire pipeline on the full dataset, saving the resulting dataframes for visualisation in a Ju[yter notebook."""

import argparse

from sm import Pipeline


def run_pipeline(args):
    """Run the entire pipeline on the full dataset, saving the resulting dataframes for visualisation in a Ju[yter notebook."""
    pipeline = Pipeline(methods=args.methods, llm_n_runs=args.llm_n_runs, use_cache=args.use_cache)
    pipeline.load_data(args.data_path)
    pipeline.extract(text_columns=args.text_columns, semicolon_columns=args.semicolon_columns)
    pipeline.analyze_geographic()
    pipeline.analyze_semantic()
    pipeline.compare_methods()
    print("\n\tAll analysis complete ready for investigation in a notebook.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", type=list, default=["nlp", "llm"])
    parser.add_argument("--llm-n-runs", type=int, default=3)
    parser.add_argument("--use-cache", type=bool, default=True)
    parser.add_argument("--data-path", type=str, default="data/EAGx_Amsterdam_11_12_25.csv")
    parser.add_argument(
        "--text-columns", type=list, default=["biography", "help_me", "help_others"]
    )
    parser.add_argument("--semicolon-columns", type=list, default=["expertise", "interests"])
    args = parser.parse_args()
    run_pipeline(args)
