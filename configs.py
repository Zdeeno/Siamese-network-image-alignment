
USUAL2 = {
    "bs": {
        "desc": None,
        "value": 80
    },
    "cs": {
        "desc": None,
        "value": 40
    },
    "fs": {
        "desc": None,
        "value": 3
    },
    "lp": {
        "desc": None,
        "value": True
    },
    "lr": {
        "desc": None,
        "value": 3.179666767281248
    },
    "nf": {
        "desc": None,
        "value": 0.25
    },
    "nm": {
        "desc": None,
        "value": "siam"
    },
    "sm": {
        "desc": None,
        "value": 5
    },
    "ech": {
        "desc": None,
        "value": 256
    },
    "res": {
        "desc": None,
        "value": 0
    },
    "_wandb": {
        "desc": None,
        "value": {
            "t": {
                "1": [
                    1
                ],
                "2": [
                    1
                ],
                "3": [
                    1,
                    16
                ],
                "4": "3.8.6",
                "5": "0.12.9",
                "8": [
                    5
                ]
            },
            "framework": "torch",
            "start_time": 1645268741,
            "cli_version": "0.12.9",
            "is_jupyter_run": False,
            "python_version": "3.8.6",
            "is_kaggle_kernel": False
        }
    }
}

BEST_PARAMS = {
    "bs": {
        "desc": None,
        "value": 96
    },
    "cs": {
        "desc": None,
        "value": 40
    },
    "fs": {
        "desc": None,
        "value": 3
    },
    "lp": {
        "desc": None,
        "value": True
    },
    "lr": {
        "desc": None,
        "value": 4.1
    },
    "nf": {
        "desc": None,
        "value": 0.3333333333333333
    },
    "nm": {
        "desc": None,
        "value": "siam_nord"
    },
    "sm": {
        "desc": None,
        "value": 2
    },
    "ech": {
        "desc": None,
        "value": 128
    },
    "res": {
        "desc": None,
        "value": 2
    },
    "_wandb": {
        "desc": None,
        "value": {
            "t": {
                "1": [
                    1
                ],
                "3": [
                    16
                ],
                "4": "3.8.6",
                "5": "0.12.9",
                "8": [
                    5
                ]
            },
            "framework": "torch",
            "start_time": 1645374059,
            "cli_version": "0.12.9",
            "is_jupyter_run": False,
            "python_version": "3.8.6",
            "is_kaggle_kernel": False
        }
    }
}

config_list = {"usual2": USUAL2, "best_params": BEST_PARAMS}


class CONFIG:

    def __init__(self, name):
        self.config = config_list[name]

    def __getattr__(self, item):
        return self.config[item]["value"]
