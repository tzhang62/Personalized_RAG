import json
import pandas as pd
import random

index_number_20 = list(range(20))

lapdog = '/Users/tzhang/projects/LAPDOG/results/valid_data_500-step_lapdog-12000.jsonl'
baseline = '/Users/tzhang/projects/LAPDOG/results/valid_data_500_baseline-step-12000.jsonl'

xl_lapdog_lst = []
with open(lapdog, 'r') as f:
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

baseline_lst = []
with open(baseline, 'r') as f:
    i=0
    for line in f:
        if i in index_number_20:
            line = json.loads(line)
            query = line['query']
            ans = str(line['answers']).strip('[]').replace("'","")
            generation1 = line['generation']
            baseline_lst.append([i,ans,generation1])
            i+=1
        else:
            i+=1
            continue

df = pd.DataFrame(xl_lapdog_lst, columns=['index_number','persona+context','answer','lapdog_pred'])
df_baseline = pd.DataFrame(baseline_lst, columns=['index_number','answer_','baseline_pred'])
df_all = df.merge(df_baseline, how='left',on='index_number')
df_all = df_all[['index_number','persona+context','answer','baseline_pred','lapdog_pred']]
df_head = df_all[['index_number','persona+context']]

all_persona = []
all_q_utters = []
all_r_utters=[]
all_ans = []
all_baseline = []
all_lapdog = []
all_index_number=[]
for i in range(len(df_all)):
    index_number = df_all['index_number'][i]
    ans = [df_all['answer'][i]]
    baseline_pred = [df_all['baseline_pred'][i]]
    lapdog_pred = [df_all['lapdog_pred'][i]]
    data = df_all['persona+context'][i]
    persona_part = data.split("context:")[0].strip()
    persona_lines = persona_part.replace("persona:","").replace("dialog with R's","").strip().split(".")
    persona=[line.strip() for line in persona_lines if line.strip()]

    # Extract the context and split into Q and R utterances
    context_part = data.split("context:")[1].strip()
    context_part = context_part.replace("<extra_id_0>","")
    lines = context_part.split("Q:")

    q_utterances = []
    r_utterances = []

    # Extract Q's and R's utterances
    for line in lines[1:]:
        q_part, *r_parts = line.split("R:")
        q_utterances.append(q_part.strip())
        if r_parts:
            r_utterances.append(r_parts[0].strip())

    diff = len(q_utterances) - len(persona)

    if diff >= 0:
        persona.extend(['']*diff)
    else:
        print('the number of diff is negative')
        import pdb;pdb.set_trace()
    num = len(q_utterances) - 1
    ans_lst = ['']*num + ans
    baseline_lst = ['']*num + baseline_pred
    lapdog_lst = ['']*num + lapdog_pred
    index_number_lst = [index_number] * len(q_utterances)
    all_index_number.append(index_number_lst)
    all_ans.append(ans_lst)
    all_baseline.append(baseline_lst)
    all_lapdog.append(lapdog_lst)
    all_persona.append(persona)
    all_q_utters.append(q_utterances)
    all_r_utters.append(r_utterances)
#import pdb; pdb.set_trace()
dataframes=[]
for i in range(len(all_index_number)):
    df_temp = pd.DataFrame({
        'index_number': all_index_number[i],
        'Persona': all_persona[i],
        'Q': all_q_utters[i],
        'R': all_r_utters[i],
        'answer': all_ans[i],
        'baseline_pred':all_baseline[i],
        'lapdog_pred': all_lapdog[i]
    })
    dataframes.append(df_temp)
final_df = pd.concat(dataframes, ignore_index=True)
#import pdb; pdb.set_trace()

def shuffle_row_columns(row):
    column_order = df_ans.columns.tolist()
    shuffled_order = column_order.copy()
    random.shuffle(shuffled_order)  # Shuffle column order
    
    # Reorder row based on shuffled column order
    shuffled_row = row[shuffled_order]
    return pd.Series([shuffled_row.tolist(), shuffled_order])

df_ans = final_df[['answer','baseline_pred','lapdog_pred']]
shuffled_results = df_ans.apply(shuffle_row_columns,axis=1)
df_head=final_df[['index_number','Persona','Q','R']]
import pdb; pdb.set_trace()


# Extract results and rebuild the shuffled DataFrame
shuffled_df = pd.DataFrame(shuffled_results[0].tolist(), columns=df_ans.columns)
shuffled_df['column_order'] = shuffled_results[1].tolist()
df_head['generation1']=shuffled_df['answer']
df_head['generation2']=shuffled_df['baseline_pred']
df_head['generation3']=shuffled_df['lapdog_pred']
df_head['column_order'] = shuffled_df['column_order']
df_head.to_excel('/Users/tzhang/Documents/lapdog_shuffled_result_new_with_order.xlsx')
import pdb; pdb.set_trace()
df_all.to_excel('/Users/tzhang/Documents/lapdog_new.xlsx')