class SingletonMeta(type):

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls in cls._instances:
            print(
                f"** Instance of {cls.__name__} already exists, returning the existing instance. **"
            )
        else:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]
