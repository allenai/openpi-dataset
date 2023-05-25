import ast
template = '{"role": "user", "content": "We have the following object attributes: {holder}. Group those that are describing the same property of the object. Add attribute names as needed. You must include the provided attribute names in the answer."}'

holder = "container of wii's av composite output"
holder = holder.replace("'", "\\'")
print(holder)

template = template.replace('{holder}', holder)
template = ast.literal_eval(template)
print(template)
