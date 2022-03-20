import json
import os
import html
from csv import DictReader
from os.path import join, isfile
from typing import List, Dict, Any


def files_from_folder(folder: str, full_paths: bool = False) -> List[str]:
    onlyfiles = [f for f in os.listdir(folder) if isfile(join(folder, f))]
    if full_paths:
        return [os.path.join(folder, filename) for filename in onlyfiles]
    else:
        return onlyfiles

def html_unescape_url(u):
    return html.unescape(u).lower().strip()

def load_img_urls(in_dir) -> Dict[str, Dict[str, Any]]:
    d: Dict[str, Dict[str, Any]] = {}
    for fp in files_from_folder(in_dir, full_paths=True):
        print(f"Loading img urls from {fp} ...")

        with open(fp) as infile:
            for csv_dict in DictReader(infile):
                url_key = html_unescape_url(csv_dict['url'])  # html.unescape("Valentine&#x27;s-Day") -> "Valentine's-Day"
                paragraph = []
                # title,url,summary,num_actions,topic,actions,actions_img,things_needed
                for step_num in range(1, int(csv_dict['num_actions']) + 1):
                    key = f"{url_key}||{step_num}"  # create unique key
                    paragraph.append(json.loads(csv_dict['actions'])[f"{step_num}"])
                    value = {"image_url": json.loads(csv_dict['actions_img'])[f"{step_num}"]}
                    d[key] = value
                d[url_key] = {"summary": csv_dict['summary'],
                              "paragraph": paragraph,
                              "things_needed": [x.strip() for x in csv_dict['things_needed'].split(",")]}
    return d


def add_img_url_to_metadata(loaded_img_dict: Dict[str, Dict[str, Any]], metadata_fp, out_fp):
    exc_count = 0
    lines = []
    with open(metadata_fp) as infile:
        for line in infile:
            lines.append(line.strip())
    with open(out_fp, 'w') as outfile:
        for line in lines:
            j = json.loads(line)
            # try:
            if "url" not in j['question_metadata']:
                correct_url = j['id']
            else:
                correct_url = f"{j['question_metadata']['url']}||{j['question_metadata']['step_id']}"
            img_url_entry = loaded_img_dict.get(html_unescape_url(correct_url))
            if img_url_entry is not None:
                j["question_metadata"]["image_url"] = img_url_entry["image_url"]
            else:
                j["question_metadata"]["image_url"] = "unknown"
                exc_count +=1
            # except Exception as exc:
            #     print(exc)
            #     print(j)
            outfile.write(json.dumps(j))
            outfile.write('\n')
        print(f"Exception count in this file {metadata_fp} = {exc_count}\n")



if __name__ == '__main__':
    d = load_img_urls(in_dir='visual/orig_hits/')
    for partition in ["train", "dev", "test"]:
        add_img_url_to_metadata(loaded_img_dict=d,
                                metadata_fp=f"data/gold_bk/{partition}/id_question_metadata.jsonl",
                                out_fp=f"data/gold/{partition}/id_question_metadata.jsonl")

