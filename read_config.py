global CONFIG
CONFIG = {}

def read_config():
    global CONFIG
    if CONFIG != {}:
        return CONFIG  # Return cached config if already read
    with open("CONFIG", 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue  # skip empty lines and comments
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            if value.isnumeric():
                value = int(value)
            CONFIG[key] = value
    return CONFIG