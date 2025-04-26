
import json
import pandas as pd

index_number_20 = list(range(20))

xl_lapdog = '/Users/tzhang/projects/LAPDOG/ckpt/xl/ckpt/valid-result.jsonl'
xl_baseline = '/Users/tzhang/projects/LAPDOG/results/evaluation_results_large_1210.json'
small_lapdog = '/Users/tzhang/projects/LAPDOG/results/valid-result-step-0.jsonl'
small_baseline = '/Users/tzhang/projects/LAPDOG/results/evaluation_results_small_1210.json'

xl_lapdog_lst = []
with open(xl_lapdog, 'r') as f:
    i=0
    for line in f:
        if i in index_number_20:
            line = json.loads(line)
            query = line['query']
            ans = str(line['answers']).strip('[]').replace("'","")
            generation1 = line['generation']
            xl_lapdog_lst.append([i,query,ans,generation1])
            i+=1
        else:
            i+=1
            continue


xl_baseline_lst = []
with open(xl_baseline,'r') as file:
    data = json.load(file)
    predictions = data['predictions']
    for index in index_number_20:
        xl_baseline_lst.append(predictions[index])

small_lapdog_lst = []
with open(small_lapdog,'r') as f:
    i=0
    for line in f:
        if i in index_number_20:
            line = json.loads(line)
            generation1 = line['generation']
            small_lapdog_lst.append(generation1)
            i+=1
        else:
            i+=1
            continue

small_baseline_lst = []
with open(small_baseline,'r') as file:
    data = json.load(file)
    predictions = data['predictions']
    for index in index_number_20:
        small_baseline_lst.append(predictions[index])

df = pd.DataFrame(xl_lapdog_lst, columns=['index_number','persona+context','answer','xl_lapdog_pred'])
df['xl_baseline_pred'] = xl_baseline_lst
df['small_lapdog_pred'] = small_lapdog_lst
df['small_baseline_pred'] = small_baseline_lst
df.to_excel('/Users/tzhang/Documents/ranking_file_ori.xlsx')



pass
            
        
