import json
import os
import fed
import re
from tqdm import tqdm
model, tokenizer = fed.load_models('microsoft/DialoGPT-large')
file_path = '/Users/tzhang/projects/LAPDOG/ckpt/xl/ckpt/valid-result.jsonl'
turn_level = ['interesting', 'engaging', 'specific', 'relevant', 'correct', 'semantically appropriate', 'understandable', 'fluent']
dialogue_level = ['coherent', 'error recovery', 'consistent', 'diverse', 'depth', 'likeable', 'understand', 'flexible', 'informative', 'inquisitive']
# Read the jsonl file and save each entry to a separate JSON file
ans_lst= []
with open(file_path, 'r') as file:
    for index, line in enumerate(tqdm(file, desc="Processing Lines", unit="lines")):
        if index < 1000:
            entry = json.loads(line)
            answers = entry['answers'][0]
            ans_lst.append(answers)

fed_result = dict()
with open(file_path, 'r') as file:
    for index, line in enumerate(tqdm(file, desc="Processing Lines", unit="lines")):
        if index < 1000:
            print(index)
            entry = json.loads(line)
            query = entry['query']
            pattern = r'context:\s*(.*)'
            match = re.search(pattern, query, re.DOTALL | re.IGNORECASE)
            if match:
                extracted_text = match.group(1).strip().replace('Q:','<|endoftext|>').replace('R:','<|endoftext|>').replace('<extra_id_0>','')
            else:
                print("'context:' not found in the text.")
            answers = entry['answers'][0]
            generation = entry['generation']
            conversation =  extracted_text + answers
            scores = fed.evaluate(conversation, model, tokenizer)
            turn_level_score = dict()
            for asp in turn_level:
                score = scores[asp]
                turn_level_score[asp] = score
            dialogue_level_score = dict()
            for asp in dialogue_level:
                score = scores[asp]
                dialogue_level_score[asp] = score
            fed_result[index] = dict()
            fed_result[index]['ans_turn_level_score'] = turn_level_score
            fed_result[index]['ans_dialogue_level_score'] = dialogue_level_score
            conversation_generation =  extracted_text + generation
            scores_generation = fed.evaluate(conversation_generation, model, tokenizer)
            generation_turn_level_score = dict()
            for asp in turn_level:
                score = scores_generation[asp]
                generation_turn_level_score[asp] = score
            generation_dialogue_level_score = dict()
            for asp in dialogue_level:
                score = scores_generation[asp]
                generation_dialogue_level_score[asp] = score
            fed_result[index]['generation_turn_level_score'] = generation_turn_level_score
            fed_result[index]['generation_dialogue_level_score'] = generation_dialogue_level_score
        
with open('fed_result_xl_lapdog.json','w') as json_file:
    json.dump(fed_result, json_file, indent=4)
import pdb; pdb.set_trace()