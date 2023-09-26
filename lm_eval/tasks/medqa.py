import numpy as np
from lm_eval.base import rf, Task
from lm_eval.metrics import mean
import torch


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
        instruction = "<s>The following is a multiple choice question about medical knowledge. Solve it in a step-by-step fashion, starting by summarizing the available information. Output a single option from the options as the final answer."
        question = doc['question']
        choice_num = "ABCDEFG"
        choices = ""
        for i in range(len(doc['choices'])):
            choices += "({}) {} ".format(choice_num[i], doc['choices'][i])
        choices = choices[:-1]

        return "{}\n\nQuestion: {}\n{}\nAnswer:".format(instruction, question, choices)
    
    def doc_to_generate(self, doc, model):
        tokenizer = model.tokenizer
        model = model.model
        # tokenizer.model_max_length = 1024
        model.config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id

        instruction = "<s>The following is a multiple choice question about medical knowledge. Solve it in a step-by-step fashion, starting by summarizing the available information. Output a single option from the options as the final answer."
        question = doc['question']
        choice_num = "ABCDEFG"
        choices = ""
        for i in range(len(doc['choices'])):
            choices += "({}) {} ".format(choice_num[i], doc['choices'][i])
        choices = choices[:-1]

        prompt = "{}\n\nQuestion: {}\n{}\nAnswer:".format(instruction, question, choices)

        tokenized_user = tokenizer.encode(f"{prompt}", add_special_tokens=False)
        with torch.no_grad():
            model_generation = model.generate(torch.tensor(tokenized_user).reshape(1, -1).cuda(), max_new_tokens=30, top_p=0.1, do_sample=True, temperature=0.7, top_k=40)[:, len(tokenized_user):]
        ans=tokenizer.batch_decode(model_generation, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

        return ans

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


class Med_QA_MC(Task):
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
        instruction = "The following is a multiple choice question about medical knowledge. Solve it in a step-by-step fashion, starting by summarizing the available information. Output a single option from the options as the final answer."
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

        lls = [
            rf.loglikelihood(ctx, " ({}) {}".format(choice_num[i], doc['choices'][i]))[0] for i in range(num_choices)
        ]
        
        return lls

    def process_results(self, doc, results):
        gold = doc['choices'].index(doc['answer'][0])
        pred = np.argmax(results)
        completion_len = np.array([float(len(i)) for i in doc["choices"]])
        acc_norm = 1.0 if np.argmax(results / completion_len) == gold else 0.0
        return {
            "acc": pred == gold,
            "acc_norm": acc_norm,
        }

    def aggregation(self):
        return {"acc": mean,
                "acc_norm": mean}

    def higher_is_better(self):
        return {"acc": True,
                "acc_norm": True}