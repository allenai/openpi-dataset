import json


def main():
    data = json.load(open('../../data/dev-data-reformatted-v4_votes_salience_1-20_hainiu.json'))

    for proc_id, proc_content in data.items():
        proc_states = proc_content['states']
        for idx, state_info in enumerate(proc_states):
            data[proc_id]['states'][idx]['global_salience'] = 0

            state_answers = state_info['answers']
            for step_id, step_info in state_answers.items():
                data[proc_id]['states'][idx]['answers'][step_id]['local_salience'] = 0

    json.dump(
        data,
        open('../../data/dev-data-reformatted-v4_votes_salience_1-20_hainiu.json', 'w'),
        indent=4
    )


if __name__ == '__main__':
    main()
