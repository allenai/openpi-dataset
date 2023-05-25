## Codes for Entity and Attribute Clustering and Canonicalization



### Entity Clustering

Codes for entity clustering are located at `./ent_code/`

1.   To run entity clustering, navigate to `./ent_code/ent_cluster/` and use the `/chat_main.py` script for ChatGPT-based clustering. Here are the arguments and example usage

     1.   [--out_path] specify the path to save the results
     2.   [--api_path] specify the path to the OpenAI API file
     3.   [--num_run] specify the number of times to run ChatGPT. If the model is run 5 times, then the result will be an aggregation of the 5 outputs.

     4.   Example Usage:

          ```python
           python chat_main.py --out_path ./results/dev_entity_chat_result_1.json --api_path ~/OpenAI.key --num_run 3
          ```

          

2.   The results are saved in `./ent_code/ent_cluster/results/`. To run evaluation, navigate to `./ent_code/` and use the `evaluate_entity.py` script. Here is an example usage

     ```python
     python evaluate_entity.py --result_dir ./ent_cluster/results/
     ```

     

---



### Attribute Clustering

Codes for attribute clustering are located at `./attr_code/`

1.   To run attribute clustering, navigate to `./attr_code/attr_cluster/` and use the `/chat_main.py` script for ChatGPT-based clustering. Here are the arguments and example usage

     1.   [--out_path] specify the path to save the results
     2.   [--api_path] specify the path to the OpenAI API file
     3.   [--num_run] specify the number of times to run ChatGPT. If the model is run 5 times, then the result will be an aggregation of the 5 outputs.
     4.   [--format] choose between _"vanilla"_ and _"concise"_. 
          1.   [vanilla]: cluster using only the attribute
          2.   [concise]: cluster using attribute in the format of {attribute of entity}

     5.   Example Usage:

          ```python
           python chat_main.py --out_path ./results/dev_attr_chat_result_1.json --api_path ~/OpenAI.key --num_run 3 --format concise
          ```

     

2.   The results are saved in `./attr_code/attr_cluster/results/`. To run evaluation, navigate to `./attr_code/` and use the `evaluate_attribute.py` script. Here is an example usage

```python
python evaluate_attribute.py --result_path ./attr_cluster/results/attr_chat_result_1.json
```

---



### Overgenerate Entities and Attributes

The above clustering code clusters existing entity and attribute annotations. The following code generate additional entity and attribute for each cluster.

1.   To overgenerating entities, navigate to `./over_generate/`. Use `entity_main.py` as follows:

     ```
     python entity_main.py --num_shot 3 --num_run 3
     ```

     -   [--num_shot] specifies the number of in-context exampels to use with ChatGPT
     -   [--num_run] specifies number of times to run ChatGPT. The results of the runs will be aggregated in the end.

2.   To overgenerate attributes, navigate to `./over_generate/` and use `attribute_main.py` as follows

     ```
     python entity_main.py --num_shot 3 --num_run 3
     ```

     -   [--num_shot] specifies the number of in-context exampels to use with ChatGPT

     -   [--num_run] specifies number of times to run ChatGPT. The results of the runs will be aggregated in the end.

The results will be saved at `~/data/data_in_new_format/dev-data-reformatted-v4-{entity/attr}-overgenerated.json`

---



### Convert Data to Use Format

To conver the original OpenPI data to the new format with canonicalization, navigate to `./utils/` and use `convert_to_new_format.py`. Use the script as follows

```
python convert_to_new_format.py
```

The new data is saved at `~/data/data_in_new_format/dev-data-reformatted-v4.json`

