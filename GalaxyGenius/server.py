"""
FastAPI server for running the application
"""

import os
import random
from typing import Tuple
import gradio as gr
import numpy as np
import pandas as pd

from htmlcodes import HTML_TEXT
import warnings
import pandas as pd
from fastapi import FastAPI
from model import (
    generate_tags,
    order_labels,
    generate_question_answer,
    generate_sentence,
)

warnings.filterwarnings("ignore")

DATA_ML = pd.read_csv("data/processed_ml_new.csv")
DATA_MTL = pd.read_csv("data/processed_mtl_new.csv")
NOBJECTS = DATA_ML.shape[0]
CUSTOM_PATH = "/gradio"

app = FastAPI()


@app.get("/")
def status():
    return {"status": "ok"}


def get_images() -> list:
    """Pick a random set of 4 images from ~37000 images.

    Returns:
        list: a list of paths for the images
    """
    nimages = 4
    img_path = "images/"
    img_paths = os.listdir(img_path)
    idx = random.sample(range(0, NOBJECTS), nimages)
    chosen_images = np.asarray(img_paths)[idx]
    full_paths = [img_path + c for c in chosen_images]
    images = [(full_paths[i], f"Image {i}") for i in range(nimages)]
    return images


def get_select_index(evt: gr.SelectData):
    return evt.index


def machine_learning(number: int, gallery: gr.Gallery) -> Tuple[str, str, str]:
    """Given the chosen index out of the four images in the gallery, we will first
    get the object name from the gallery and send a post request to the server. This
    will then return the different components of our infrastructure.

    Args:
        number (int): the index chosen
        gallery (gr.Gallery): the gallery from the Gradio library

    Returns:
        Tuple[str, str, str]: the tags, the multi-task learning labels, and the generated sentence.
    """
    obj_name = gallery[number][0].split(os.sep)[-1][:-4]

    # multilabel case (taggings)
    test_obj = DATA_ML[DATA_ML["newnames"] == obj_name]
    tags = generate_tags(test_obj.iloc[0])
    tags_out = "".join([f"{t}\n" for t in tags])

    # multitask case
    test_obj_mtl = DATA_MTL[DATA_MTL["newnames"] == obj_name].iloc[0, 3:]
    new_tree, new_test = order_labels(test_obj_mtl)
    mtl_out = generate_question_answer(new_test)
    mtl_ques_ans = ""

    for i, (k, v) in enumerate(mtl_out.items()):
        mtl_ques_ans += f"{i+1}) " + k + "\n" + v + "\n\n"

    # generate sentence
    sentence = generate_sentence(new_tree, new_test)

    response = {
        "object": obj_name,
        "tags": tags_out,
        "mtl": mtl_ques_ans,
        "sentence": sentence,
    }

    return response["tags"], response["mtl"], response["sentence"]


with gr.Blocks() as demo:
    gr.HTML(HTML_TEXT)

    gallery = gr.Gallery(
        label="Your Images",
        columns=[2],
        rows=[2],
        min_width=200,
        interactive=False,
        height=700,
    )
    btn = gr.Button("Get Your Images", scale=0)
    btn.click(get_images, None, gallery)
    selected = gr.Number(label="Image Selected", show_label=True)
    index_selected = gallery.select(get_select_index, None, selected)

    gr.Interface(
        machine_learning,
        [selected, gallery],
        [
            gr.Textbox(label="Tags"),
            gr.Textbox(label="Classifier"),
            gr.Textbox(label="NLP Model"),
        ],
    )

gr.mount_gradio_app(app, demo, path=CUSTOM_PATH)
