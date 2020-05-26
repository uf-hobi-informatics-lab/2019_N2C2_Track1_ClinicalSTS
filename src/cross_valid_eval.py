# -*- coding: utf-8 -*-
############################
#   evaluate 5-fold cv on clinical dataset result
############################

import argparse
import os
import sys
from pathlib import Path
import logging


def main(model, results_dir, eval_output_dir):
    logger = logging.getLogger("cv_result_eval")
    # model = "bert-large"
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    # create result file & dir
    # eval_output_dir = "../output/{}_5f_eval/".format(model)
    Path(eval_output_dir).mkdir(exist_ok=True, parents=True)
    # read all results
    # results_dir = "/home/ma.yingha/workspace/py3/clinicalsts/output/{}/tmp".format(
    #     model)
    dirs = os.listdir(results_dir)

    results = "\n"
    best_pearson = 0.0
    best_epoch = 0
    best_batch = 0
    for d in dirs:
        logger.info("processing %s", d)
        [batch_size, epoch_num] = d.split("_")
        logger.info("batch size {}, epoch num {}".format(
            batch_size, epoch_num))

        # iterate through 5 samples
        ls, pr = .0, .0

        for i in range(5):
            result_file = os.path.join(
                results_dir, d, "sample{}".format(i), "eval_results.txt")
            with open(result_file, "r") as f:
                for line in f.readlines():
                    if line.startswith("pearson"):
                        pr += float(line.split("=")[-1].strip())
                    elif line.startswith("eval_loss"):
                        ls += float(line.split("=")[-1].strip())
        # take the average of loss and pearson correlation
        hyper = str(batch_size) + "_" + str(epoch_num)
        avg_ls = ls / 5.0
        avg_pr = pr / 5.0
        # curr_result = "hyper" + hyper + "avg_loss" + str(avg_ls) + "avg_pearson" + str(avg_pr) + "\n"
        curr_result = "hypyer = %s\tavg_loss = %s\tavg_pearson = %s\n" % (
            hyper, str(avg_ls), str(avg_pr))
        if avg_pr > best_pearson:
          best_pearson = avg_pr
          best_epoch = epoch_num
          best_batch = batch_size
        results += curr_result

    results += f"Best epoch: {best_epoch}\nBest batch size: {best_batch}\nBest score: {best_pearson}"
    logger.info(results)
    eval_output = os.path.join(eval_output_dir, "5f-result.txt")
    with open(eval_output, "w") as f:
        f.write(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_type",
        default=None,
        required=True,
        help="Type of model you are currently training"
    )
    parser.add_argument(
        "--input_dir",
        default=None,
        required=True,
        help="Directory that contains all the results after 5-fold cross validation"
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        required=True,
        help="The output dir for the result"
    )

    args = parser.parse_args()
    main(args.model_type, args.input_dir, args.output_dir)
