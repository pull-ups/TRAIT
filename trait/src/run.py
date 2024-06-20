import time, json, sys, os, torch, argparse
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI


from util.option_dict_4 import *
from util.prompts import  get_prompt
from util.lm_format import apply_format


def save_json(file_name, res_arr):
    with open(file_name, 'w') as f:
        json.dump(res_arr, f, indent=4, ensure_ascii=False)
        
device = "cuda"



def get_likelihood(model, input_ids):
    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids)
    logits = outputs.logits[:, -1, :]  # Logits for the last token
    probabilities = torch.softmax(logits, dim=-1)
    return probabilities

def chatgpt_response(client, query=''):
    while True:
        try:
            chat_completion = client.chat.completions.create(
                messages = [
                    {'role':'user', 'content':query},],
                model="gpt-3.5-turbo-0125",
                logprobs=True,
                top_logprobs=20,
                max_tokens=1,
                temperature=0.0,
            )
            return chat_completion
        except Exception as e:
            print(e)
            time.sleep(5)
            continue
        

def gpt4_response(client, query=''):
    while True:
        try:
            chat_completion = client.chat.completions.create(
                messages = [
                    {'role':'user', 'content':query},],
                model="gpt-4-turbo-2024-04-09",
                logprobs=True,
                top_logprobs=20,
                max_tokens=1,
                temperature=0.0,
            )
            return chat_completion
        except Exception as e:
            print(e)
            time.sleep(5)
            continue

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=None, required=True)
    parser.add_argument('--model_name_short', type=str, default=None, required=True)
    parser.add_argument('--inference_type', type=str, default="base")
    parser.add_argument('--prompt_type', type=int, default=1)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--paraphrase', action='store_true')
    return parser.parse_args()


def main():
    args = get_args()
    print(f"python {' '.join(sys.argv)}")
    
    if "gpt" not in args.model_name_short.lower():
        model = AutoModelForCausalLM.from_pretrained(args.model_name, load_in_8bit=True, device_map='cuda', trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    elif "gpt" in args.model_name_short.lower():
        client = OpenAI(
            api_key="Your Key"
        )
    
    data=json.load(open("../../TRAIT.json"))
    
    if args.paraphrase:
        run_type="inference_likelihood_paraphrase"
    else:
        run_type="inference_likelihood"
    subdir=f"prompt_type_{args.prompt_type}"
    save_dir=f"../{run_type}/{subdir}"
    
    save_file_dir=os.path.join(save_dir, f"results_option_{args.model_name_short}.json")
    print("save_dir", save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
        
        
    res_arr=[]
    for idx, sample in enumerate(data):
        print(idx)
        if args.paraphrase:
            instruction=sample["paraphrased_situation"]+" "+sample["paraphrased_query"]
        else:
            instruction=sample["situation"]+" "+sample["query"]
        response_high1=sample["response_high1"]
        response_high2=sample["response_high2"]
        response_low1=sample["response_low1"]
        response_low2=sample["response_low2"]
        
        for rev in [False, True]:
            prompt=get_prompt(args.prompt_type, rev, instruction, response_high1, response_high2, response_low1, response_low2)


            if "gpt" in args.model_name_short.lower():
                vocab_probabilities={}
                if args.model_name_short=="Chatgpt":
                    response=chatgpt_response(client, prompt)
                elif args.model_name_short=="gpt4":
                    response=gpt4_response(client, prompt)
                output_response=response.choices[0].message.content
                logprobs=response.choices[0].logprobs.content
                logprobs_at_0=logprobs[0].top_logprobs
                for item in logprobs_at_0:
                    token=item.token
                    logprob=item.logprob
                    vocab_probabilities[token]=np.exp(logprob)
            else:
                encoded=apply_format(prompt, args.inference_type, tokenizer)
                likelihoods = get_likelihood(model, encoded).squeeze().tolist()
                vocab_probabilities={}
                
                if args.prompt_type==1:
                    option_tokens=get_option_token("ABCD")
                elif args.prompt_type==2:
                    option_tokens=get_option_token("1234")
                elif args.prompt_type==3:
                    option_tokens=get_option_token("ABCD")
                for token in option_tokens:
                    vocab_probabilities[token]=likelihoods[tokenizer.convert_tokens_to_ids(token)]
                vocab_probabilities = dict(sorted(vocab_probabilities.items(), key=lambda item: item[1], reverse=True))
                vocab_probabilities = {k: vocab_probabilities[k] for k in list(vocab_probabilities)[:10]}

            if rev:
                sample[f"prompt_rev"]=prompt
                sample[f"likelihood_rev"]=vocab_probabilities
            else:
                sample[f"prompt"]=prompt
                sample[f"likelihood"]=vocab_probabilities
                
                
        res_arr.append(sample)
        if len(res_arr)%args.save_interval==0:
            save_json(save_file_dir, res_arr)
    save_json(save_file_dir, res_arr)
        
        
if __name__ == '__main__':
    main()

#python run.py --model_name Chatgpt --model_name_short Chatgpt --prompt_type 1
#python run.py --model_name mistralai/Mistral-7B-Instruct-v0.2 --model_name_short mistral_instruct --inference_type chat --prompt_type 1