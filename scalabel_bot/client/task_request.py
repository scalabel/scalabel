from typing import Dict, List


def get_opt_item() -> List[Dict[str, object]]:
    return [
        {
            "prompt": "Paris is the capital city of France.",
            "length": 100,
        },
    ]


def get_opt_items() -> List[Dict[str, object]]:
    return [
        {
            "prompt": "Paris is the capital city of France.",
            "length": 100,
        },
        {
            "prompt": "Computer science is the study of computation and",
            "length": 100,
        },
        {
            "prompt": "The University of California, Berkeley is a public",
            "length": 100,
        },
        {
            "prompt": (
                "Ion Stoica is a Romanian-American computer"
                " scientist specializing in"
            ),
            "length": 100,
        },
        {
            "prompt": "Today is a good day and I want to",
            "length": 100,
        },
        {
            "prompt": "What is the valuation of Databricks?",
            "length": 100,
        },
        {
            "prompt": "Which country has the most population?",
            "length": 100,
        },
        {
            "prompt": "What do you think about the future of Cryptocurrency?",
            "length": 100,
        },
        {
            "prompt": "What do you think about the meaning of life?",
            "length": 100,
        },
        {
            "prompt": "Donald Trump is the president of",
            "length": 100,
        },
    ]


def get_image_item() -> List[Dict[str, object]]:
    return [
        {
            "attributes": {},
            "intrinsics": {
                "center": [771.31406, 360.79945],
                "focal": [1590.83437, 1592.79032],
            },
            "labels": [],
            "name": "bot-batch",
            "sensor": -1,
            "timestamp": -1,
            "url": "https://s3-us-west-2.amazonaws.com/scalabel-public/demo/synscapes/img/rgb/1.png",
            "videoName": "",
        }
    ]
