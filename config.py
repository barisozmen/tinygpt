class Config:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def set(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)