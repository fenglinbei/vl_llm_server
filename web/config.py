latex_delimiters_set = [
        {"left": "$$", "right": "$$", "display": True},
        {"left": "$", "right": "$", "display": False},
        {"left": "\\(", "right": "\\)", "display": False},
        {"left": "\\[", "right": "\\]", "display": True},
    ]

class Config:
    def __init__(self) -> None:
        self.user_avatar = "default"
        self.bot_avatar = "default"


config = Config()