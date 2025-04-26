import json
import numpy as np
import pandas as pd
###xl lapdog model analysis
xl_lapdog = '/Users/tzhang/projects/LAPDOG/ckpt/xl/ckpt/fed_result_small_LAPDOG_1213.json'
ans_turn_level = dict()
ans_dialogue_level = dict()
pred_turn_level = dict()
pred_dialogue_level = dict()
with open(xl_lapdog,'r') as file:
    predicted = json.load(file)
    for index in predicted.keys():
        for asp in predicted[index]['ans_turn_level_score']:
            ans_turn_level[asp] = []
            ans_turn_level[asp].append(predicted[index]['ans_turn_level_score'][asp])
            pred_turn_level[asp] = []
            pred_turn_level[asp].append(predicted[index]['generation_turn_level_score'][asp])
        for asp in predicted[index]['generation_dialogue_level_score']:
            ans_dialogue_level[asp] = []
            ans_dialogue_level[asp].append(predicted[index]['ans_dialogue_level_score'][asp])
            pred_dialogue_level[asp] = []
            pred_dialogue_level[asp].append(predicted[index]['generation_dialogue_level_score'][asp])
results_turn_level = []
for asp in ans_turn_level:
    results_turn_level.append([asp,np.mean(ans_turn_level[asp]),np.mean(pred_turn_level[asp]), np.mean(ans_turn_level[asp])-np.mean(pred_turn_level[asp])])
results_dialogue_level = []
for asp in ans_dialogue_level:
    results_dialogue_level.append([asp,np.mean(ans_dialogue_level[asp]),np.mean(pred_dialogue_level[asp]),np.mean(ans_dialogue_level[asp])-np.mean(pred_dialogue_level[asp])])

df_turn_level = pd.DataFrame(results_turn_level, columns = ['aspects','ans','pred','difference'])
sum_row = pd.DataFrame(df_turn_level.sum(axis=0)).T
sum_row['aspects'] = ['Total']
df_turn_level = pd.concat([df_turn_level, sum_row])
df_dialogue_level = pd.DataFrame(results_dialogue_level, columns = ['aspects','ans','pred','difference'])
sum_row_dia = pd.DataFrame(df_dialogue_level.sum(axis=0)).T
sum_row_dia['aspects']=['Total']
df_dialogue_level =pd.concat([df_dialogue_level, sum_row_dia])
import pdb; pdb.set_trace()

df_turn_level.to_excel('/Users/tzhang/Documents/fed_result_new_lapdog_small_turn.xlsx')
df_dialogue_level.to_excel('/Users/tzhang/Documents/fed_result_new_lapdog_small_dia.xlsx')