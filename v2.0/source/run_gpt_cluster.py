import openai
openai.api_key = "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"

entity_prompt = [
    {"role": "system", "content": "You are a helpful assistant that groups objects together."},
    {"role": "user", "content": "We have the following objects: ingredients, dough, dough pieces, carob fruit balls, balls, mixing bowl, spoon, plate, mixture, carob, hand, bowl. Group those that refer to the same thing."},
    {"role": "assistant", "content": """The objects that refer to the same things are:
- ingredients
- dough, dough pieces
- carob fruit balls, balls, carob
- mixing bowl, bowl
- spoon
- plate
- mixture
- hand
"""},
    {"role": "user", "content": "We have the following objects: the blackboard eraser, eraser, glovebox, eraser shelf at the store, windshield, inside windshield, bank account, glove box, money, hands. Group those that refer to the same thing."},
]

attribute_prompt = [
    {"role": "system", "content": "You are a helpful assistant that groups attributes that descirbe objects."},
    {"role": "user", "content": "We have the following attributes: composition, consistency, shape, size, readiness, location, hardness, temperature, volume, weight, number, density, fullness, stickiness"},
    {"role": "assistant", "content": """The objects that refer to the same things are:
- ingredients
- dough, dough pieces
- carob fruit balls, balls, carob
- mixing bowl, bowl
- spoon
- plate
- mixture
- hand
"""},
    {"role": "user", "content": "We have the following objects: the blackboard eraser, eraser, glovebox, eraser shelf at the store, windshield, inside windshield, bank account, glove box, money, hands. Group those that refer to the same thing."},
]

def run_chatgpt(entity_prompt, model="gpt-3.5-turbo", temperature=0.0):
    ret = openai.ChatCompletion.create(
        model=model,
        messages=entity_prompt
        )
    gen_text = dict(ret["choices"][0]["message"])
    return gen_text

gen_text = run_chatgpt(entity_prompt)
print(gen_text)