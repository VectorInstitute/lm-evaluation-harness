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


class MMLU_Professional_Med(Task):
    VERSION = 0
    DATASET_PATH = "lukaemon/mmlu"
    DATASET_NAME = "professional_medicine"

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
        instruction = "<s>[INST] The following is a multiple choice question about medical knowledge. Solve it in a step-by-step fashion, starting by summarizing the available information from the abstract. Output a single option from the four options as the final answer."
        question = doc['input']
        choices = "(A) {} (B) {} (C) {} (D) {}".format(doc['A'], doc['B'], doc['C'], doc['D'])

        return "{}\n\nQuestion: {}\n{}\nAnswer: [/INST]".format(instruction, question, choices)

    def doc_to_generate(self, doc, model):
        tokenizer = model.tokenizer
        model = model.model
        # tokenizer.model_max_length = 1024
        model.config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id

        instruction = "<s>[INST] The following is a multiple choice question about medical knowledge. Solve it in a step-by-step fashion, starting by summarizing the available information from the abstract. Output a single option from the four options as the final answer."
        question = doc['input']
        choices = "(A) {} (B) {} (C) {} (D) {}".format(doc['A'], doc['B'], doc['C'], doc['D'])

        prompt = "{}\n\nQuestion: {}\n{}\nAnswer: [/INST]".format(instruction, question, choices)

        tokenized_user = tokenizer.encode(f"{prompt}", add_special_tokens=False)
        with torch.no_grad():
            model_generation = model.generate(torch.tensor(tokenized_user).reshape(1, -1).cuda(), max_new_tokens=500, top_p=0.1, do_sample=True, temperature=0.7, top_k=40)[:, len(tokenized_user):]
        ans=tokenizer.batch_decode(model_generation, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

        return ans

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["question"] + " " + "\n".join(doc["context"]["contexts"])

    def doc_to_target(self, doc):
        return " ({})".format(doc['target'])

    def construct_requests(self, doc, ctx):
        """Uses RequestFactory to construct Requests and returns
        an iterable of Requests which will be sent to the LM.
        """
    
        ll_A, _ = rf.loglikelihood(ctx, " (A)")
        ll_B, _ = rf.loglikelihood(ctx, " (B)")
        ll_C, _ = rf.loglikelihood(ctx, " (C)")
        ll_D, _ = rf.loglikelihood(ctx, " (D)")
        return ll_A, ll_B, ll_C, ll_D
      

    def process_results(self, doc, results):
        gold = doc["target"]
        pred = np.argmax(results)
        return {
            "acc": ["A", "B", "C", "D"][pred] == gold,
        }

    def aggregation(self):
        return {"acc": mean}

    def higher_is_better(self):
        return {"acc": True}