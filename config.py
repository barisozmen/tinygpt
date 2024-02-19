

class Config:
    def set(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


_the_config = None

def the_config():
    global _the_config
    if _the_config is None:
        _the_config = Config()
    return _the_config