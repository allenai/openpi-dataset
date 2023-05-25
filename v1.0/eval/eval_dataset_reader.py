import json

# This class reads an OpenPI dataset folder
# e.g. data/gold/dev/
# such folder will always contain 4 jsonl files
# id_question.jsonl, id_answers.jsonl, id_question_metadata.jsonl, and id_answers_metadata.jsonl
# This class reads and stores all the information and gives out easy access functions like
# get_question_for_id, get_question_metadata_for_id ...


class DatasetReader:

    def __init__(self, in_path: str):
        self.in_path = in_path
        self.id_to_key_to_record_dict = dict()
        self.load_dataset(in_path=in_path)

    def load_file(self, file_fp: str, key: str):
        with open(file_fp, 'r') as open_file:
            for line in open_file:
                if not line.strip():
                    continue
                record = json.loads(line)
                id = record["id"]
                key_record_dict = self.id_to_key_to_record_dict.get(id, dict())
                key_record_dict[key] = record[key]
                self.id_to_key_to_record_dict[id] = key_record_dict

    def load_dataset(self, in_path: str):
        for key in ["question", "answers", "question_metadata", "answers_metadata"]:
            in_fp = f"{in_path}/id_{key}.jsonl"
            self.load_file(file_fp=in_fp, key=key)

    def __get_value(self, id: str, val_key, default_val):
        return self.id_to_key_to_record_dict.get(id, dict()).get(val_key, default_val)

    def get_num_questions(self):
        return len(self.id_to_key_to_record_dict)

    def get_all_question_ids(self):
        return self.id_to_key_to_record_dict.keys()

    def get_all_question_ids_by_qfilter(self, question_filter: dict):
        filtered_ids = []

        for id, record in self.id_to_key_to_record_dict.items():
            keep_record = True
            if 'question_metadata' not in record:
                raise Exception("No question metadata found")

            # Check if metadata field matches with any of the target list in the question filter
            question_metadata = record["question_metadata"]
            for f_key, f_vals in question_filter.items():
                if question_metadata[f_key] not in f_vals:
                    keep_record = False
                    break
            if keep_record:
                filtered_ids.append(id)

        return filtered_ids

    def get_question_for_id(self, id: str):
        return self.__get_value(id=id, val_key="question", default_val="")

    def get_question_metadata_for_id(self, id: str):
        return self.__get_value(id=id, val_key="question_metadata", default_val=dict())

    def get_answers_for_id(self, id: str, answer_filter={}):
        if not answer_filter:
            return self.__get_value(id=id, val_key="answers", default_val=[])

        if 'answers_metadata' not in self.id_to_key_to_record_dict[id]:
            raise Exception("No answer metadata found")
        filtered_answers = []
        for answer_metadata in self.id_to_key_to_record_dict[id]['answers_metadata']:
            keep_answer = True
            for f_key, f_vals in answer_filter.items():
                if answer_metadata[f_key] not in f_vals:
                    keep_answer = False
                    break
            if keep_answer:
                filtered_answers.append(answer_metadata["answer"])
        return filtered_answers

    def get_answers_metadata_for_id(self, id: str):
        return self.__get_value(id=id, val_key="answers_metadata", default_val=dict())


class QuestionPredictionsDirReader(DatasetReader):
    def load_dataset(self, in_path: str):
        for key in ["question", "question_metadata", "answers", "answers_metadata"]:
            in_fp = f"{in_path}/id_{key}.jsonl"
            self.load_file(file_fp=in_fp, key=key)


class PredictionsDirReader(DatasetReader):

    def load_dataset(self, in_path: str):
        for key in ["answers", "answers_metadata"]:
            in_fp = f"{in_path}/id_{key}.jsonl"
            self.load_file(file_fp=in_fp, key=key)


class SingleFileReader(DatasetReader):
    def __init__(self, in_path: str, type: str):
        self.url_to_stepsEntries = dict()
        self.type = type
        super().__init__(in_path=in_path)

    def load_dataset(self, in_path: str):
        for key in [self.type]:
            self.load_file(file_fp=in_path, key=key)

    def load_file(self, file_fp: str, key: str):
        prev_url = ""
        steps = []
        with open(file_fp, 'r') as open_file:
            for line in open_file:
                if not line.strip():
                    continue
                record = json.loads(line)
                id = record["id"]
                key_record_dict = self.id_to_key_to_record_dict.get(id, dict())
                key_record_dict[key] = record[key]
                self.id_to_key_to_record_dict[id] = key_record_dict
                q_id_parts = id.split('||')
                url = q_id_parts[0]
                if url != prev_url:
                    self.url_to_stepsEntries[prev_url] = steps
                    prev_url = url
                    steps = []  # reset steps.
                steps.append(record)
        # add last record into the dictionary
        self.url_to_stepsEntries[prev_url] = steps


class QuestionFileReader(SingleFileReader):
    def __init__(self, in_path: str):
        super().__init__(in_path=in_path, type="question")


class PredictionsFileReader(SingleFileReader):
    def __init__(self, in_path: str):
        super().__init__(in_path=in_path, type="answers")


class QuestionMetaFileReader(SingleFileReader):
    def __init__(self, in_path: str):
        super().__init__(in_path=in_path, type="question_metadata")


class AnswersMetaFileReader(SingleFileReader):
    def __init__(self, in_path: str):
        super().__init__(in_path=in_path, type="answers_metadata")

