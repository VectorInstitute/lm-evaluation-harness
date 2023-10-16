import json
import torch

base_a = -8.802260398864746
base_b = -10.43479061126709
base_c = -10.180068016052246
base_d = -11.16988754272461

base_chat_a = -8.324846267700195 # Answer: (A)
base_chat_b =  -9.646142959594727
base_chat_c = -9.43743896484375
base_chat_d = -10.430400848388672

base_chat_2_a = -10.417497634887695 # The answer to the question is (A)
base_chat_2_b = -10.4735746383667
base_chat_2_c = -10.505844116210938
base_chat_2_d = -9.505159378051758

ft_a = -6.058724880218506
ft_b =  -6.866930961608887
ft_c = -7.060182094573975
ft_d = -7.703822135925293

# 1. Check data path.
# 2. check base_a.
# 3. check gt_answer. origin -> results[i]['truth'] if not, use augmented.

datasets = ['mmlu_anatomy', 'mmlu_clinical_knowledge', 'mmlu_college_bio', 'mmlu_college_med', 'mmlu_medical_genetics', 'mmlu_professional_med']
choices = ['A', 'B', 'C', 'D']

for dataset in datasets:
    for choice in choices:
        with open("/scratch/ssd004/scratch/ywchoi/{}_write_out_info_7b_base_chat_0toNone_{}.json".format(dataset, choice)) as f:
            results = json.load(f)

        total = len(results)
        correct = 0

        for i in range(len(results)):
            a_prob = torch.exp(torch.tensor(results[i]['logit_0']))
            b_prob = torch.exp(torch.tensor(results[i]['logit_1']))
            c_prob = torch.exp(torch.tensor(results[i]['logit_2']))
            d_prob = torch.exp(torch.tensor(results[i]['logit_3']))

            unconditional_a_prob = a_prob / torch.exp(torch.tensor(base_chat_a))
            unconditional_b_prob = b_prob / torch.exp(torch.tensor(base_chat_b))
            unconditional_c_prob = c_prob / torch.exp(torch.tensor(base_chat_c))
            unconditional_d_prob = d_prob / torch.exp(torch.tensor(base_chat_d))

            predicted = torch.argmax(torch.tensor([unconditional_a_prob, unconditional_b_prob, unconditional_c_prob, unconditional_d_prob]))
            gt_answer = results[i]['truth']
            predicted_answer = [" (A)", " (B)", " (C)", " (D)"]
            # if predicted_answer[predicted] == gt_answer:
            if predicted_answer[predicted] == " ({})".format(choice):
                correct +=1
        print(dataset, choice, correct/total)

# with open("/scratch/ssd004/scratch/ywchoi/mmlu_anatomy_write_out_info_7b_base_chat_0toNone_A.json") as f:
#     results = json.load(f)

# total = len(results)
# correct = 0

# for i in range(len(results)):
#     a_prob = torch.exp(torch.tensor(results[i]['logit_0']))
#     b_prob = torch.exp(torch.tensor(results[i]['logit_1']))
#     c_prob = torch.exp(torch.tensor(results[i]['logit_2']))
#     d_prob = torch.exp(torch.tensor(results[i]['logit_3']))

#     unconditional_a_prob = a_prob / torch.exp(torch.tensor(base_chat_a))
#     unconditional_b_prob = b_prob / torch.exp(torch.tensor(base_chat_b))
#     unconditional_c_prob = c_prob / torch.exp(torch.tensor(base_chat_c))
#     unconditional_d_prob = d_prob / torch.exp(torch.tensor(base_chat_d))

#     predicted = torch.argmax(torch.tensor([unconditional_a_prob, unconditional_b_prob, unconditional_c_prob, unconditional_d_prob]))
#     gt_answer = results[i]['truth']
#     predicted_answer = [" (A)", " (B)", " (C)", " (D)"]
#     # if predicted_answer[predicted] == gt_answer:
#     if predicted_answer[predicted] == " (A)":
#         correct +=1
# print(correct/total)
