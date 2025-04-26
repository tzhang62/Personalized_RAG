import pandas as pd
import ast
import numpy as np
from sacrebleu.metrics import BLEU
from src.evaluation import rouge_score, f1_score, normalize_answer

rank_file = '/Users/tzhang/Documents/LAPDOG_project/Ranking_all.xlsx'
df_rank = pd.read_excel(rank_file)
order_file = '/Users/tzhang/Documents/lapdog_shuffled_result_new_with_order.xlsx'
df_order = pd.read_excel(order_file) 

index_lst = df_rank['index_number'].unique()

for index in index_lst:
    df_individual = df_rank[df_rank['index_number'] == index]
    
generated_dict = {}
generated_dict['lapdog_pred'] = []
generated_dict['baseline_pred'] = []
generated_dict['answer'] = []
for index in index_lst:
    string_lst = df_order[df_order['index_number']==index].reset_index()['column_order'][len(df_order[df_order['index_number']==index])-1]
    order_lst = ast.literal_eval(string_lst)
    df_temp = df_rank[df_rank['index_number'] == index]
    generated_dict[order_lst[0]].append(df_temp['generation1'].reset_index(drop=True)[len(df_temp)-1].strip())
    generated_dict[order_lst[1]].append(df_temp['generation2'].reset_index(drop=True)[len(df_temp)-1].strip())
    generated_dict[order_lst[2]].append(df_temp['generation3'].reset_index(drop=True)[len(df_temp)-1].strip())

def calc_bleu(targets, pred_text):
    # Create a list of reference lists, where each prediction has its corresponding target
    references = [[target] for target in targets]
    bleu = BLEU().corpus_score(pred_text, references)
    return bleu

lapdog_pred = generated_dict['lapdog_pred']
targets = generated_dict['answer']
baseline_pred = generated_dict['baseline_pred']
#import pdb; pdb.set_trace()

bleu_lapdog = calc_bleu(targets, lapdog_pred)
bleu_baseline = calc_bleu(targets, baseline_pred)
print(bleu_lapdog.format())
print(bleu_baseline.format())

def calc_rouge(pred_text, targets):

    rouge = [rouge_score(pred, [target]) for pred, target in zip(pred_text, targets)]
    rouge_mat = np.asmatrix(rouge)*100
    averaged = rouge_mat.mean(0).tolist()[0]
    return averaged

def calc_f1(pred_text, targets):


    score = [f1_score(pred, [target], normalize_answer) for pred, target in zip(pred_text, targets)]
    avg = np.asarray(score).mean()*100
    return avg


rouge_lapdog = calc_rouge(lapdog_pred, targets)
rouge_baseline = calc_rouge(baseline_pred, targets)
print(rouge_lapdog)
print(rouge_baseline)

f1_lapdog = calc_f1(lapdog_pred, targets)
f1_baseline = calc_f1(baseline_pred, targets)
print(f1_lapdog)
print(f1_baseline)
#import pdb; pdb.set_trace()

# First calculate all metrics
lapdog_metrics = {
    'F1': f1_lapdog,
    'BLEU': bleu_lapdog.score,  # Note: BLEU object has .score attribute
    'ROUGE-L': rouge_lapdog[2]  # Assuming rouge returns [rouge1, rouge2, rougeL]
}

baseline_metrics = {
    'F1': f1_baseline,
    'BLEU': bleu_baseline.score,
    'ROUGE-L': rouge_baseline[2]
}

# Calculate improvements (relative change)
improvements = {
    'F1↑': ((f1_lapdog - f1_baseline) / f1_baseline) * 100,
    'BLEU↑': ((bleu_lapdog.score - bleu_baseline.score) / bleu_baseline.score) * 100,
    'ROUGE-L↑': ((rouge_lapdog[2] - rouge_baseline[2]) / rouge_baseline[2]) * 100
}

# Calculate average improvement
improvements['AVG↑'] = sum([improvements['F1↑'], improvements['BLEU↑'], improvements['ROUGE-L↑']]) / 3

# Print formatted table
print("| Model | F1 | BLEU | ROUGE-L | F1↑ | BLEU↑ | ROUGE-L↑ | AVG↑ |")
print("|--------|-----|------|----------|------|--------|-----------|------|")
print(f"| Baseline | {baseline_metrics['F1']:.2f} | {baseline_metrics['BLEU']:.2f} | {baseline_metrics['ROUGE-L']:.2f} | 0.00% | 0.00% | 0.00% | 0.00% |")
print(f"| LAPDOG | {lapdog_metrics['F1']:.2f} | {lapdog_metrics['BLEU']:.2f} | {lapdog_metrics['ROUGE-L']:.2f} | {improvements['F1↑']:.2f}% | {improvements['BLEU↑']:.2f}% | {improvements['ROUGE-L↑']:.2f}% | {improvements['AVG↑']:.2f}% |")