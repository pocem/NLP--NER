import json
import re
import os
import matplotlib.pyplot as plt
import numpy as np

OVERLAP_THRESHOLD = 0.5


def clean_llm_json(raw_str):
    """Extract JSON list from messy LLM output"""
    try:
        match = re.search(r'\[.*\]', raw_str, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return []
    except:
        return []


def get_gt_data_structures(sample):
    """
    Returns:
        strict_set = (entity,label)
        text_set = entity only
    """

    strict_set = set()
    text_set = set()

    for ent in sample.get("entities", []):

        text_list = ent.get("text", [""])
        text = str(text_list[0]).lower().strip()

        labels = ent.get("semantic_type_id", [])

        if text:

            text_set.add(text)

            for l in labels:
                strict_set.add((text, l))

    return strict_set, text_set


def overlap_score(pred, gt):

    pred_words = set(pred.split())
    gt_words = set(gt.split())

    inter = len(pred_words & gt_words)
    union = len(pred_words | gt_words)

    return inter / union if union else 0


def relaxed_match(pred_pairs, gt_pairs):
    """
    Dimension 3:
    overlap + label
    """

    matched_gt = set()

    tp = 0

    for pred_ent, pred_label in pred_pairs:

        for gt_ent, gt_label in gt_pairs:

            if gt_label != pred_label:
                continue

            if (gt_ent, gt_label) in matched_gt:
                continue

            score = overlap_score(
                pred_ent,
                gt_ent
            )

            if score >= OVERLAP_THRESHOLD:

                tp += 1
                matched_gt.add(
                    (gt_ent, gt_label)
                )

                break

    fp = len(pred_pairs) - tp
    fn = len(gt_pairs) - tp

    return tp, fp, fn


def relaxed_entity_match(
        pred_texts,
        gt_texts
):
    """
    Dimension 4:
    overlap only
    """

    matched_gt = set()

    tp = 0

    for pred_ent in pred_texts:

        for gt_ent in gt_texts:

            if gt_ent in matched_gt:
                continue

            score = overlap_score(
                pred_ent,
                gt_ent
            )

            if score >= OVERLAP_THRESHOLD:

                tp += 1
                matched_gt.add(gt_ent)

                break

    fp = len(pred_texts) - tp
    fn = len(gt_texts) - tp

    return tp, fp, fn


def calc_metrics(tp, fp, fn):

    precision = (
        tp/(tp+fp)
        if (tp+fp) > 0
        else 0
    )

    recall = (
        tp/(tp+fn)
        if (tp+fn) > 0
        else 0
    )

    f1 = (
        2*precision*recall /
        (precision+recall)

        if (precision+recall) > 0
        else 0
    )

    return (
        precision,
        recall,
        f1
    )




def evaluate_all_dimensions():

    gt_path = r"C:\Users\misog\SCHOOL\4th semester\Natural language processing\project\Segmenting\test.json"

    pred_path = r"C:\Users\misog\SCHOOL\4th semester\Natural language processing\project\project\llm_outputs\llama70B_full_oneshot.json"

    output_txt_path = "llm_result_oneshot.txt"

    try:

        with open(
            gt_path,
            "r",
            encoding="utf-8"
        ) as f:

            gt_data = {
                sample["pmid"]: sample
                for sample in json.load(f)
            }

        with open(
            pred_path,
            "r",
            encoding="utf-8"
        ) as f:

            pred_data = json.load(f)

    except FileNotFoundError as e:

        print(e)
        return


    # dimension 1
    tp_strict = fp_strict = fn_strict = 0

    # dimension 2
    tp_text = fp_text = fn_text = 0

    # dimension 3
    tp_relaxed = fp_relaxed = fn_relaxed = 0

    # dimension 4
    tp_relaxed_entity = 0
    fp_relaxed_entity = 0
    fn_relaxed_entity = 0


    with open(
        output_txt_path,
        "w",
        encoding="utf-8"
    ) as out_f:


        header = (
            f"{'PMID':<15}|"
            f"{'GT':<6}|"
            f"{'PRED':<6}|"
            f"{'StrictTP':<10}|"
            f"{'TextTP':<10}|"
            f"{'RelaxTP':<10}|"
            f"{'RelaxEnt':<10}\n"
        )

        separator = (
            "-"*90+"\n"
        )

        out_f.write(
            header
        )

        out_f.write(
            separator
        )

        print(
            header+
            separator
        )


        for pred_item in pred_data:

            pmid = str(
                pred_item["pmid"]
            )

            if pmid not in gt_data:
                continue


            (
                gt_strict,
                gt_texts

            ) = get_gt_data_structures(
                gt_data[pmid]
            )


            pred_json = clean_llm_json(
                pred_item[
                    "raw_output"
                ]
            )


            pred_strict = set()

            pred_texts = set()


            for item in pred_json:

                entity = (
                    str(
                        item.get(
                            "entity",
                            ""
                        )
                    )
                    .lower()
                    .strip()
                )

                labels = item.get(
                    "labels",
                    []
                )


                if entity:

                    pred_texts.add(
                        entity
                    )

                    for l in labels:

                        pred_strict.add(
                            (
                                entity,
                                l
                            )
                        )


            ################################################
            # STRICT
            ################################################

            curr_tp_strict = len(
                gt_strict.intersection(
                    pred_strict
                )
            )

            tp_strict += (
                curr_tp_strict
            )

            fp_strict += len(
                pred_strict -
                gt_strict
            )

            fn_strict += len(
                gt_strict -
                pred_strict
            )


            ################################################
            # ENTITY EXACT
            ################################################

            curr_tp_text = len(
                gt_texts.intersection(
                    pred_texts
                )
            )

            tp_text += (
                curr_tp_text
            )

            fp_text += len(
                pred_texts -
                gt_texts
            )

            fn_text += len(
                gt_texts -
                pred_texts
            )


            ################################################
            # RELAXED + LABEL
            ################################################

            (
                curr_tp_relaxed,
                curr_fp_relaxed,
                curr_fn_relaxed

            ) = relaxed_match(
                pred_strict,
                gt_strict
            )


            tp_relaxed += (
                curr_tp_relaxed
            )

            fp_relaxed += (
                curr_fp_relaxed
            )

            fn_relaxed += (
                curr_fn_relaxed
            )


            ################################################
            # RELAXED ENTITY ONLY
            ################################################

            (
                curr_tp_relaxed_ent,
                curr_fp_relaxed_ent,
                curr_fn_relaxed_ent

            ) = relaxed_entity_match(
                pred_texts,
                gt_texts
            )


            tp_relaxed_entity += (
                curr_tp_relaxed_ent
            )

            fp_relaxed_entity += (
                curr_fp_relaxed_ent
            )

            fn_relaxed_entity += (
                curr_fn_relaxed_ent
            )


            row = (
                f"{pmid:<15}|"
                f"{len(gt_texts):<6}|"
                f"{len(pred_texts):<6}|"
                f"{curr_tp_strict:<10}|"
                f"{curr_tp_text:<10}|"
                f"{curr_tp_relaxed:<10}|"
                f"{curr_tp_relaxed_ent:<10}\n"
            )

            out_f.write(
                row
            )

            print(
                row,
                end=""
            )


    prec_s,rec_s,f1_s = calc_metrics(
        tp_strict,
        fp_strict,
        fn_strict
    )

    prec_t,rec_t,f1_t = calc_metrics(
        tp_text,
        fp_text,
        fn_text
    )

    prec_r,rec_r,f1_r = calc_metrics(
        tp_relaxed,
        fp_relaxed,
        fn_relaxed
    )

    prec_re,rec_re,f1_re = calc_metrics(
        tp_relaxed_entity,
        fp_relaxed_entity,
        fn_relaxed_entity
    )
    

    summary = (

        "\n"
        + "="*60 + "\n"
        + "FINAL NER PERFORMANCE REPORT\n"
        + "="*60 + "\n\n"

        + "[DIMENSION 1]\n"
        + "STRICT NER\n"
        + "Exact entity + exact label\n"
        + "-"*40 + "\n"

        + f"Precision: {prec_s:.4f}\n"
        + f"Recall:    {rec_s:.4f}\n"
        + f"F1 Score:  {f1_s:.4f}\n\n"


        + "[DIMENSION 2]\n"
        + "ENTITY EXTRACTION\n"
        + "Exact entity only\n"
        + "-"*40 + "\n"

        + f"Precision: {prec_t:.4f}\n"
        + f"Recall:    {rec_t:.4f}\n"
        + f"F1 Score:  {f1_t:.4f}\n\n"


        + "[DIMENSION 3]\n"
        + "RELAXED NER\n"
        + "Overlap + correct label\n"
        + "-"*40 + "\n"

        + f"Precision: {prec_r:.4f}\n"
        + f"Recall:    {rec_r:.4f}\n"
        + f"F1 Score:  {f1_r:.4f}\n\n"


        + "[DIMENSION 4]\n"
        + "RELAXED ENTITY EXTRACTION\n"
        + "Overlap only\n"
        + "-"*40 + "\n"

        + f"Precision: {prec_re:.4f}\n"
        + f"Recall:    {rec_re:.4f}\n"
        + f"F1 Score:  {f1_re:.4f}\n"

        + "="*60
    )


    with open(
        output_txt_path,
        "a",
        encoding="utf-8"
    ) as out_f:

        out_f.write(
            summary
        )


    print(summary)

    print(
        "\nCombined plot saved."
    )

    

if __name__=="__main__":
    evaluate_all_dimensions()