"""
Predictions from our trained Deep Learning models.
"""

from typing import Tuple
import pandas as pd
import torch
from transformers import T5Tokenizer


TOKENIZER = T5Tokenizer.from_pretrained("t5-base")

if torch.cuda.is_available():
    DEV = "cuda"
else:
    DEV = "cpu"

# the model
MODEL = torch.load(
    "models/t5-model-finetuned-50-epochs.bin", map_location=torch.device(DEV)
)
MODEL.eval()

QUESTIONS = {
    "task_1": "Is the galaxy simply smooth and rounded, with no sign of a disk?",
    "task_2": "How rounded is it?",
    "task_3": "Could this be a disk viewed edge-on?",
    "task_4": "Is the galaxy merging or disturbed?",
    "task_5": "Does the galaxy have a bulge at its centre? If so, what shape?",
    "task_6": "Is there a bar feature through the centre of the galaxy?",
    "task_7": "Is there any sign of a spiral arm pattern?",
    "task_8": "How tightly wound do the spiral arms appear?",
    "task_9": "How many spiral arms are there?",
    "task_10": "Is there a central bulge? Is so, how large is it compared with the galaxy?",
}

TASK_MAPPING = {
    "task_1": {
        "Smooth": "smooth",
        "Featured or Disk": "has features or disk",
        "Artifact": "artifact",
    },
    "task_2": {
        "Round": "round",
        "In Between": "elliptical",
        "Cigar Shaped": "cigar-shaped",
    },
    "task_3": {
        "Edge On Disk (Yes)": "has an edge-on disk",
        "Edge On Disk (No)": "does not have an edge-on disk",
    },
    "task_4": {
        "Merging (Merger)": "merging",
        "Merging (Major Disturbance)": "merging with major disturbance",
        "Merging (Minor Disturbance)": "merging with minor disturbance",
        "Merging (None)": "not merging",
    },
    "task_5": {
        "Bulge (Rounded)": "rounded central bulge",
        "Bulge (Boxy)": "boxy central bulge",
        "Bulge (None)": "no central bulge",
    },
    "task_6": {"No Bar": "no bar", "Weak Bar": "weak bar", "Strong Bar": "strong bar"},
    "task_7": {
        "Spiral Arms (Yes)": "has spiral arms",
        "Spiral Arms (No)": "does not have spiral arms",
    },
    "task_8": {
        "Spiral Winding (Tight)": "tight spiral winding",
        "Spiral Winding (Medium)": "medium spiral winding",
        "Spiral Winding (Loose)": "loose spiral winding",
    },
    "task_9": {
        "Spiral Arms (1)": "one spiral arm",
        "Spiral Arms (2)": "two spiral arms",
        "Spiral Arms (3)": "three spiral arms",
        "Spiral Arms (4)": "four spiral arms",
        "Spiral Arms (More Than 4)": "more than four spiral arms",
        "Spiral Arms (cannot tell)": "no spiral arms",
    },
    "task_10": {
        "Central Bulge (None)": "no central bulge",
        "Central Bulge (Small)": "small central bulge",
        "Central Bulge (Moderate)": "moderate central bulge",
        "Central Bulge (Large)": "large central bulge",
        "Central Bulge (Dominant)": "dominant central bulge",
    },
}


def generate_tags(ml_labels: pd.Series) -> list:
    """Output the tags of the object given a pandas series which contains 0 and 1.

    Args:
        ml_labels (pd.Series): the pandas series with the tag names.

    Returns:
        list: a list of tags corresponding to that object.
    """

    return list(ml_labels.index[ml_labels == 1])


def order_labels(mtl_labels: pd.Series) -> Tuple[str, pd.Series]:
    """Given a pandas series which contains the classes of object,
    we will reorder it such that it follows the same tree structure
    as in the Galaxy Zoo notation.

    Args:
        mtl_labels (pd.Series): the pandas series

    Returns:
        Tuple[str, pd.Series]: the tree and the reordered tasks.
    """

    t_1 = ["task_1", "task_2", "task_4"]
    t_2 = ["task_1", "task_3", "task_5", "task_4"]
    t_3 = ["task_1", "task_3", "task_6", "task_7", "task_10", "task_4"]
    t_4 = [
        "task_1",
        "task_3",
        "task_6",
        "task_7",
        "task_8",
        "task_9",
        "task_10",
        "task_4",
    ]
    t_5 = ["task_1"]
    trees = [t_1, t_2, t_3, t_4, t_5]

    tasks = list(mtl_labels.index[~pd.isnull(mtl_labels)])
    for i, tree in enumerate(trees):
        if len(tasks) == 1:
            return f"tree_5", mtl_labels[t_5]
        else:
            criterion = all([t in tree for t in tasks])
            if criterion:
                return f"tree_{i+1}", mtl_labels[tree]


def generate_question_answer(mtl_labels: pd.Series) -> dict:
    """Given a pandas series which contains the different tasks,
    we will return a dictionary with the question and the answer.

    Args:
        mtl_labels (pd.Series): the pandas series with the tasks and answer.

    Returns:
        dict: a dictionary with the question and answer.
    """
    tasks = list(mtl_labels.index)
    record = {}
    for i, t in enumerate(tasks):
        record[QUESTIONS[t]] = mtl_labels.values[i]
    return record


def prompt_generate(keywords: str) -> str:
    """Generate a prompt based on a set of keywords in the following format:

    f"{word_1} | {word_2}"

    Args:
        keywords (str): a set of words defined above.

    Returns:
        str: the generated sentence from the T5 model.
    """
    input_ids = TOKENIZER.encode(
        keywords + "</s>", max_length=512, truncation=True, return_tensors="pt"
    )
    input_ids = input_ids.to(DEV)
    outputs = MODEL.generate(input_ids, do_sample=True, max_length=1024)
    output_text = TOKENIZER.decode(outputs[0])
    return output_text[6:-4]


def tree_1(label_1: str, label_2: str, label_3: str) -> str:
    """Generates a sentence based on the labels of the first tree.

    Args:
        label_1 (str): label in Task 1
        label_2 (str): label in Task 2
        label_3 (str): label in Task 4

    Returns:
        str: the generated sentence from the NLP model.
    """
    newlabel_1 = TASK_MAPPING["task_1"][label_1]
    newlabel_2 = TASK_MAPPING["task_2"][label_2]
    newlabel_3 = TASK_MAPPING["task_4"][label_3]
    keywords = f"{newlabel_1} | {newlabel_2} | {newlabel_3}"
    return prompt_generate(keywords)


def tree_2(label_1: str, label_2: str, label_3: str, label_4: str) -> str:
    """Generates a sentence based on the labels of the second tree.

    Args:
        label_1 (str): label in Task 1
        label_2 (str): label in Task 3
        label_3 (str): label in Task 5
        label_4 (str): label in Task 4

    Returns:
        str: the generated sentence from the NLP model.
    """
    newlabel_1 = TASK_MAPPING["task_1"][label_1]
    newlabel_2 = TASK_MAPPING["task_3"][label_2]
    newlabel_3 = TASK_MAPPING["task_5"][label_3]
    newlabel_4 = TASK_MAPPING["task_4"][label_4]
    keywords = f"{newlabel_1} | {newlabel_2} | {newlabel_3} | {newlabel_4}"
    return prompt_generate(keywords)


def tree_3(
    label_1: str, label_2: str, label_3: str, label_4: str, label_5: str, label_6: str
) -> str:
    """Generates a sentence based on the labels of the third tree.

    Args:
        label_1 (str): label in Task 1
        label_2 (str): label in Task 3
        label_3 (str): label in Task 6
        label_4 (str): label in Task 7
        label_5 (str): label in Task 10
        label_6 (str): label in Task 4

    Returns:
        str: the generated sentence from the NLP model.
    """
    newlabel_1 = TASK_MAPPING["task_1"][label_1]
    newlabel_2 = TASK_MAPPING["task_3"][label_2]
    newlabel_3 = TASK_MAPPING["task_6"][label_3]
    newlabel_4 = TASK_MAPPING["task_7"][label_4]
    newlabel_5 = TASK_MAPPING["task_10"][label_5]
    newlabel_6 = TASK_MAPPING["task_4"][label_6]
    keywords = f"{newlabel_1} | {newlabel_2} | {newlabel_3} | {newlabel_4} | {newlabel_5} | {newlabel_6}"
    return prompt_generate(keywords)


def tree_4(
    label_1: str,
    label_2: str,
    label_3: str,
    label_4: str,
    label_5: str,
    label_6: str,
    label_7: str,
    label_8: str,
) -> str:
    """Generates a sentence based on the labels of the fourth tree.

    Args:
        label_1 (str): label in Task 1
        label_2 (str): label in Task 3
        label_3 (str): label in Task 6
        label_4 (str): label in Task 7
        label_5 (str): label in Task 8
        label_6 (str): label in Task 9
        label_7 (str): label in Task 10
        label_8 (str): label in Task 4

    Returns:
        str: the generated sentence from the NLP model.
    """
    newlabel_1 = TASK_MAPPING["task_1"][label_1]
    newlabel_2 = TASK_MAPPING["task_3"][label_2]
    newlabel_3 = TASK_MAPPING["task_6"][label_3]
    newlabel_4 = TASK_MAPPING["task_7"][label_4]
    newlabel_5 = TASK_MAPPING["task_8"][label_5]
    newlabel_6 = TASK_MAPPING["task_9"][label_6]
    newlabel_7 = TASK_MAPPING["task_10"][label_7]
    newlabel_8 = TASK_MAPPING["task_4"][label_8]
    keywords = f"{newlabel_1} | {newlabel_2} | {newlabel_3} | {newlabel_4} | {newlabel_5} | {newlabel_6} | {newlabel_7} | {newlabel_8}"
    return prompt_generate(keywords)


def tree_5(label_1: str) -> str:
    """This is just an artifact in the decision tree.

    Args:
        label_1 (str): the label (artifact)

    Returns:
        str: the generated sentence from the NLP model.
    """
    newlabel_1 = TASK_MAPPING["task_1"][label_1]
    keywords = f"{newlabel_1}"
    return prompt_generate(keywords)


def generate_sentence(treename: str, labels: pd.Series) -> str:
    """Given the tree name and the multi-task learning (ordered) labels,
    we then generate a sentence using the pre-trained model.

    Args:
        treename (str): the tree name: one of [tree_1, tree_2, tree_3, tree_4, tree_5]
        labels (pd.Series): the ordered labels for the multi-task learning problem.

    Returns:
        str: the generated sentence from the NLP model.
    """
    if treename == "tree_1":
        sentence = tree_1(*labels.values)

    if treename == "tree_2":
        sentence = tree_2(*labels.values)

    if treename == "tree_3":
        sentence = tree_3(*labels.values)

    if treename == "tree_4":
        sentence = tree_4(*labels.values)

    if treename == "tree_5":
        sentence = tree_5(*labels.values)
    return sentence
