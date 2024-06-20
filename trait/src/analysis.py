import os, json, sys, argparse
import numpy as np
import pandas as pd
from util.option_dict_4 import *








def get_score(cnt_dict, ):
    personality_arr=["Agreeableness", "Conscientiousness", "Extraversion", "Neuroticism", "Openness", "Psychopathy", "Machiavellianism", "Narcissism"]
    score_arr=[]
    for personality in personality_arr:
        if (cnt_dict[personality]["high"]+cnt_dict[personality]["low"])==0:
            print("continue")
            continue
        score=( (cnt_dict[personality]["high"]) / (cnt_dict[personality]["high"]+cnt_dict[personality]["low"]) )*100
        
        score_arr.append(score)
    return score_arr







def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=None, required=True)
    parser.add_argument('--prompt_type', type=int, default=1)
    return parser.parse_args()


personality_arr=["Agreeableness", "Conscientiousness", "Extraversion", "Neuroticism", "Openness", "Psychopathy", "Machiavellianism", "Narcissism"]

cnt_dict={
    "Agreeableness": {"high":0, "low":0},
    "Conscientiousness": {"high":0, "low":0},
    "Extraversion": {"high":0, "low":0},
    "Neuroticism": {"high":0, "low":0},
    "Openness": {"high":0, "low":0},
    "Psychopathy": {"high":0, "low":0},
    "Machiavellianism": {"high":0, "low":0},
    "Narcissism": {"high":0, "low":0},
}



def main():
    args = get_args()
    data=json.load(open(f"../inference_likelihood/prompt_type_{args.prompt_type}/results_option_{args.model_name}.json"))

    if args.prompt_type==1:
        option_tokens=get_option_token("ABCD")
    elif args.prompt_type==2:
        option_tokens=get_option_token("1234")
    elif args.prompt_type==3:
        option_tokens=get_option_token("ABCD")

    for i, sample in enumerate(data):
        personality=sample["personality"]
        if "gpt" in args.model_name.lower():
            likelihood_A=np.exp(sample["likelihood"][option_tokens[0]]) if option_tokens[0] in sample["likelihood"] else 0
            likelihood_B=np.exp(sample["likelihood"][option_tokens[1]]) if option_tokens[1] in sample["likelihood"] else 0
            likelihood_C=np.exp(sample["likelihood"][option_tokens[2]]) if option_tokens[2] in sample["likelihood"] else 0
            likelihood_D=np.exp(sample["likelihood"][option_tokens[3]]) if option_tokens[3] in sample["likelihood"] else 0
            
            likelihood_A_rev=np.exp(sample["likelihood_rev"][option_tokens[0]]) if option_tokens[0] in sample["likelihood_rev"] else 0
            likelihood_B_rev=np.exp(sample["likelihood_rev"][option_tokens[1]]) if option_tokens[1] in sample["likelihood_rev"] else 0
            likelihood_C_rev=np.exp(sample["likelihood_rev"][option_tokens[2]]) if option_tokens[2] in sample["likelihood_rev"] else 0
            likelihood_D_rev=np.exp(sample["likelihood_rev"][option_tokens[3]]) if option_tokens[3] in sample["likelihood_rev"] else 0
        else:
            likelihood_A=sample["likelihood"][option_tokens[0]] if option_tokens[0] in sample["likelihood"] else 0
            likelihood_B=sample["likelihood"][option_tokens[1]] if option_tokens[1] in sample["likelihood"] else 0
            likelihood_C=sample["likelihood"][option_tokens[2]] if option_tokens[2] in sample["likelihood"] else 0
            likelihood_D=sample["likelihood"][option_tokens[3]] if option_tokens[3] in sample["likelihood"] else 0
            
            likelihood_A_rev=sample["likelihood_rev"][option_tokens[0]] if option_tokens[0] in sample["likelihood_rev"] else 0
            likelihood_B_rev=sample["likelihood_rev"][option_tokens[1]] if option_tokens[1] in sample["likelihood_rev"] else 0
            likelihood_C_rev=sample["likelihood_rev"][option_tokens[2]] if option_tokens[2] in sample["likelihood_rev"] else 0
            likelihood_D_rev=sample["likelihood_rev"][option_tokens[3]] if option_tokens[3] in sample["likelihood_rev"] else 0
        likelihood_A_norm=likelihood_A/(likelihood_A+likelihood_B+likelihood_C+likelihood_D)
        likelihood_B_norm=likelihood_B/(likelihood_A+likelihood_B+likelihood_C+likelihood_D)
        likelihood_C_norm=likelihood_C/(likelihood_A+likelihood_B+likelihood_C+likelihood_D)
        likelihood_D_norm=likelihood_D/(likelihood_A+likelihood_B+likelihood_C+likelihood_D)
        
        likelihood_A_rev_norm=likelihood_A_rev/(likelihood_A_rev+likelihood_B_rev+likelihood_C_rev+likelihood_D_rev)
        likelihood_B_rev_norm=likelihood_B_rev/(likelihood_A_rev+likelihood_B_rev+likelihood_C_rev+likelihood_D_rev)
        likelihood_C_rev_norm=likelihood_C_rev/(likelihood_A_rev+likelihood_B_rev+likelihood_C_rev+likelihood_D_rev)
        likelihood_D_rev_norm=likelihood_D_rev/(likelihood_A_rev+likelihood_B_rev+likelihood_C_rev+likelihood_D_rev)
        
        high_1=(likelihood_A_norm+likelihood_B_rev_norm)/2
        low_1=(likelihood_B_norm+likelihood_A_rev_norm)/2
        high_2=(likelihood_C_norm+likelihood_D_rev_norm)/2
        low_2=(likelihood_D_norm+likelihood_C_rev_norm)/2
        max_option=np.argmax([high_1, high_2, low_1, low_2])
        if max_option in [0, 1]:
            cnt_dict[personality]["high"]+=1
        elif max_option in [2, 3]:
            cnt_dict[personality]["low"]+=1
    score_arr=get_score(cnt_dict)
    for personality, score in zip(personality_arr, score_arr):
        print(f"{personality}: {score}")
    


if __name__ == '__main__':
    main()