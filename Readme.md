#  TRAIT: Personality Testset designed for LLMs with Psychometrics
> **Do LLMs Have Distinct and Consistent Personalities? TRAIT: Personality Testset designed for LLMs with Psychometrics**,
Seungbeen Lee<sup>1,*</sup>,
Seungwon Lim<sup>1,*</sup>,
Seungju Han<sup>2,3</sup>, 
Giyeong Oh<sup>1</sup>, 
Hyungjoo Chae<sup>1</sup>,
Jiwan Chung<sup>1</sup>
Minju Kim<sup>1</sup>
Beong-woo Kwak<sup>1</sup>
Yeonsoo Lee<sup>4</sup>
Dongha Lee<sup>1</sup>
Jinyoung Yeo<sup>1</sup>
Youngjae Yu<sup>1</sup><br>
<sup>1</sup>Yonsei University,
<sup>2</sup>Seoul National University 
<sup>3</sup>Allen Institute for AI
<sup>4</sup>NCSOFT<br>
<sup>\*</sup>Equal contribution.


​
## Preparation
You do not need a complex environment.
Install the required python packages with the following command:
​
```
pip install -r requirements.txt
```
​
## How to run

You can run 2 mode: Base mode and Personality-prompt mode. With Base mode,

### Base mode
Chatgpt
```
python run.py --model_name Chatgpt --model_name_short Chatgpt --prompt_type 1
```
mistralai/Mistral-7B-Instruct-v0.2
```
python run.py --model_name mistralai/Mistral-7B-Instruct-v0.2 --model_name_short mistral_instruct --inference_type chat --prompt_type 1
```


### Personality-prompt mode
Chatgpt
```
python run_personality_prompt.py --model_name Chatgpt --model_name_short Chatgpt --model_close --personality "high openness" --prompt_type 2
```
mistralai/Mistral-7B-Instruct-v0.2
```
python run_personality_prompt.py --model_name mistralai/Mistral-7B-Instruct-v0.2 --model_name_short mistral_instruct --inference_type chat --personality "high openness" --prompt_type 2
```


### Result
You can get the result of the model. 
```
python analysis.py
```

**More codes are available soon!**