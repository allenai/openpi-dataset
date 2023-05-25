import os 
import json


def main():
    data = json.load(open('../../data/dev-data-reformatted-v4_votes_salience_1-20_hainiu.json'))

    counter = 0
    for proc_id, proc_content in data.items():
        if int(proc_id) <= 5:
            counter += 1
            continue

        if counter >= 20:
            break 
        else:
            counter += 1

        goal = proc_content['goal']
        steps = proc_content['steps']
        steps = '\n'.join([f'[Step{i+1}] {content}' for (i, content) in enumerate(steps)])
        proc_states = proc_content['states']
        for idx, state_info in enumerate(proc_states):
            os.system('clear')
            print('------------------------------------------------')
            print(f'Proc Progress: {proc_id}/20 ')
            print(f'Entity Progress: {idx}/{len(state_info)} ')
            print('------------------------------------------------')
            print(f'Goal: {goal}')
            print('------------------------------------------------')
            print(steps)
            print('------------------------------------------------\n')
            
            current_entity = state_info['entity']
            print(f'Current Entity: {current_entity}')
            print('------------------------------------------------\n')
            data[proc_id]['states'][idx]['global_salience'] = input('Input Global Saliency:\n')
            print('------------------------------------------------\n')

            state_answers = state_info['answers']
            for step_id, step_info in state_answers.items():
                data[proc_id]['states'][idx]['answers'][step_id]['local_salience'] = input(f'Input Local Saliency for {step_id}:\n')
    
    with open('../../data/hainiu_cached/dev-data-reformatted-v4_votes_salience_1-20_hainiu.json', 'w') as f:
        json.dump(data, f, indent=4)
    f.close()

if __name__ == '__main__':
    main()
