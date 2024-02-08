# API for OpenPI2.0-related utitlties


## Overview
Input: A list of procedures, each consisting of a goal and a list of steps.
Output:
1. The predicted schema (entities and corresponding attributes) of each step
2. The predicted states based on the schema above of each step
3. The predicted global entity salience of the entire procedure 
4. The predicted local entity salience of each step
Note that 1 and 2 use `text-davinci-003` while 3 and 4 use `gpt-3.5-turbo` which are the best models for these tasks according to the paper.

## Usage
1. Format your input as `trial.json`. 
2. Set your OpenAI API key to the path in the `openai.api_key =` line in `predict_all.py`, or change that line to your desired path.
3. Run `predict_all.py --input INPUT_PATH --output OUTPUT_PATH` produces the output JSON file. 