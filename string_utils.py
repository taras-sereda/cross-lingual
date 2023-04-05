import uuid


def validate_and_preprocess_title(title: str) -> str:
    if not isinstance(title, str):
        raise Exception(f"Title must be an instance of str, received {title} of type {type(title)}")
    title = title.strip()
    if len(title) == 0:
        raise Exception(f"Title can't be an empty string")
    return title


def get_random_string():
    return str(uuid.uuid4())
