import logging

from src.options import Options
from src.tasks.base import BaseTask, filter_results_by_id

logger = logging.getLogger(__name__)
import json


class Task(BaseTask):

    def __init__(self, opt: Options, *args, **kwargs):
        self.mt = opt.mt
        if self.mt:
            self.labels = json.load(open(opt.label_file))

    def process(self, example, *args, **kwargs):

        clean_input = example["query"]

        clean_target = example["target"]

        if not "passages" in example:
            example["passages"] = [{"title": "", "text": ""}]

        example["query"] = clean_input
        example["target"] = clean_target
        example["metadata"] = {}
        example["metadata"]["id"] = example["id"]

        # for mt setting
        if self.mt:
            label = self.labels[str(example["id"])]
            example["label"] = label
        else:
            example["label"] = 3

        return example

    def filter(self, *args, **kwargs):
        """Remove the passage we are trying to generate from retrieved results"""
        return filter_results_by_id(*args, **kwargs)
