import openai

prompt = [
    {"role": "system", "content": "You are a helpful assistant that figures out involved entities and attributes in procedures."},
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

def run_chatgpt(prompt, model="gpt-3.5-turbo", temperature=0.0):
    ret = openai.ChatCompletion.create(
        model=model,
        messages=prompt,
        temperature=temperature,
        )
    gen_text = dict(ret["choices"][0]["message"])
    return gen_text

gen_text = run_chatgpt(prompt)
print(gen_text)