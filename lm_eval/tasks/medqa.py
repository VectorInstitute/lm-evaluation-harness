import numpy as np
from lm_eval.base import rf, Task
from lm_eval.metrics import mean


_CITATION = """
@inproceedings{jin2019pubmedqa,
    title={PubMedQA: A Dataset for Biomedical Research Question Answering},
    author={Jin, Qiao and Dhingra, Bhuwan and Liu, Zhengping and Cohen, William and Lu, Xinghua},
    booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
    pages={2567--2577},
    year={2019}
}
"""


class Med_QA(Task):
    VERSION = 0
    DATASET_PATH = "bigbio/med_qa"
    DATASET_NAME = "med_qa_en_bigbio_qa"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def test_docs(self):
        if self.has_test_docs():
            # HF is labelled as train but its really just for testing
            return self.dataset["test"]

    def doc_to_text(self, doc):
        instruction = "The following is a multiple choice question about medical knowledge. Solve it in a step-by-step fashion, starting by summarizing the available information. Output a single option from the four options as the final answer."
        question = doc['question']
        choice_num = "ABCDEFG"
        choices = ""
        for i in range(len(doc['choices'])):
            choices += "({}) {} ".format(choice_num[i], doc['choices'][i])
        choices = choices[:-1]

        return "Instruction: {}\n\nQuestion: {}\n{}\nAnswer:".format(instruction, question, choices)

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["question"] + " " + "\n".join(doc["context"]["contexts"])

    def doc_to_target(self, doc):
        choice_num = "ABCDEFG"
        choice = doc['choices'].index(doc['answer'][0])

        return " ({})".format(choice_num[choice])

    def construct_requests(self, doc, ctx):
        """Uses RequestFactory to construct Requests and returns
        an iterable of Requests which will be sent to the LM.
        """
        choice_num = "ABCDEFG"
        num_choices = len(doc['choices'])

        if num_choices == 1:
            ll_A, _ = rf.loglikelihood(ctx, " (A)")
            return ll_A
        elif num_choices == 2:
            ll_A, _ = rf.loglikelihood(ctx, " (A)")
            ll_B, _ = rf.loglikelihood(ctx, " (B)")
            return ll_A, ll_B
        elif num_choices == 3:
            ll_A, _ = rf.loglikelihood(ctx, " (A)")
            ll_B, _ = rf.loglikelihood(ctx, " (B)")
            ll_C, _ = rf.loglikelihood(ctx, " (C)")
            return ll_A, ll_B, ll_C
        elif num_choices == 3:
            ll_A, _ = rf.loglikelihood(ctx, " (A)")
            ll_B, _ = rf.loglikelihood(ctx, " (B)")
            ll_C, _ = rf.loglikelihood(ctx, " (C)")
            ll_D, _ = rf.loglikelihood(ctx, " (D)")
            return ll_A, ll_B, ll_C, ll_D
        else:
            ll_A, _ = rf.loglikelihood(ctx, " (A)")
            ll_B, _ = rf.loglikelihood(ctx, " (B)")
            ll_C, _ = rf.loglikelihood(ctx, " (C)")
            ll_D, _ = rf.loglikelihood(ctx, " (D)")
            ll_E, _ = rf.loglikelihood(ctx, " (E)")
            return ll_A, ll_B, ll_C, ll_D, ll_E

    def process_results(self, doc, results):
        gold = doc['choices'].index(doc['answer'][0])
        pred = np.argmax(results)
        return {
            "acc": pred == gold,
        }

    def aggregation(self):
        return {"acc": mean}

    def higher_is_better(self):
        return {"acc": True}