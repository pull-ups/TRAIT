import json

def get_system_prompt(personality):
    personality_description_data=json.load(open("../../personality_prompt_BFI.json"))
    personality_description_dict={}
    for personality in personality_description_data["behavior"]:
        for hl in ["high", "low"]:
            key=f"{hl} {personality}"
            personality_description_dict[key]=personality_description_data["behavior"][personality][hl]
    for key in personality_description_dict:
        desc_str=""
        for i, desc in enumerate(personality_description_dict[key]):
            desc_str+=f"{i+1}. {desc} "
        personality_description_dict[key]=desc_str
        
        
    system_prompt=f"You are an assistant with {personality}. Following statements are descriptions of {personality}.\n{personality_description_dict[personality]}"
    return system_prompt                