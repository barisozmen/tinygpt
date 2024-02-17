_the_config = None

class Config:
    def set(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

def the_config():
    global _the_config
    if _the_config is None:
        _the_config = Config()
    return _the_config