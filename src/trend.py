import pandas as pd
from ipywidgets import (
    Box,
    Button,
    HBox,
    IntText,
    Layout,
    SelectMultiple,
    Textarea,
    VBox,
)
from main import SCA


class StructuralCollocationAnalysis(SCA):
    def __init__(self, *args, **kwargs):
        SCA.__init__(self, *args, **kwargs)

        self.colloc_selector = SelectMultiple(
            opions={f"{_[0]}-{_[1]}": _ for _ in self.collocates},
            rows=len(self.collocates),
            description="Select Collocates",
        )

        self.colloc_writer = Textarea(description="Enter new collocates:")

        self.window_selector = IntText(description="Window size:", value=10)

        # add metadata selector

        self.button_calc_collocates = Button(
            description="Add collocates",
        )
        self.button_calc_collocates.on_click(self.create_collocates)

        self.button_count_collocates = Button(
            description="count collocates",
        )
        self.button_calc_collocates.on_click(self.count_collocates)

        line0 = [
            self.colloc_writer,
            HBox(self.button_calc_collocates, self.window_selector),
        ]
        line1 = [self.buttom_calc_collocates]

        line2 = [self.colloc_selector]

        box_layout = Layout(
            display="flex", flex_flow="row", align_items="stretch"
        )

        self.vbox = VBox(
            [
                Box(line0, layout=box_layout),
                Box(line1, layout=box_layout),
                Box(line2, layout=box_layout),
            ]
        )

        self.last_query = None

    def text_to_collocate(self):
        for line in self.colloc_writer.split("\n"):
            pair = line.split()
            if len(pair) == 2:
                yield pair

    def create_collocates(self):
        self.add_collocates(self.text_to_collocate())

    def load_col_options(self):
        self.colloc_selector.options = {
            f"{_[0]}-{_[1]}": _ for _ in self.collocates
        }

    def count_collocates(self):
        selected_collocates = set(self.colloc_selector.get_interact_value())
        selected_window = self.window_selector.value

        current_query = (selected_collocates, selected_window)

        if current_query != self.last_query:
            self.df = self.count_with_collocates(
                (p1, p2, selected_window) for p1, p2 in selected_collocates
            )
