import requests
from ast import literal_eval


def get_response(text):
    url = "http://127.0.0.1:5000/v1/completions"
    headers = {
        "Content-Type": "application/json"
    }

    params = {
        "prompt": text,
        "max_tokens": 200,
        "temperature": 0.1,
        "typical_p": 1,
        "seed": 10,
        "return_full_text": False,
        "repetition_penalty": 1.15,
        "repetition_penalty_range":1024,
        "guidance_scale": 1,
        "skip_special_tokens": False
    }

    response = requests.post(url, headers=headers, json=params)

    answer = literal_eval(response.text)['choices'][0]['text']
    #answer = response.text
    return answer

prompt = 'Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n### Instruction:\nGiven a sentence, extract "chemical" entities from it by highlighting them with <mark> and </mark>. If not present, output the same sentence. No further explanation needed.\n### Input:\nHowever , pretreatment with 5 , 10 and 20 mg / kg i . p . trazodone enhanced dexamphetamine stereotypy , and antagonized haloperidol catalepsy , ergometrine - induced WDS behavior and fluoxetine - induced penile erections . \n\n### Response:\n'

get_response(prompt)