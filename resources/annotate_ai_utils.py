from pathlib import Path

def file_contents(path):
    """
    Get the OpenAI API key from the given path
    :param path:
    :return:
    """
    key_path = Path(path).expanduser()
    return key_path.read_text().strip()


def read_guideline(guideline_path):
    """
    Read the guideline from the given path
    :param guideline_path:
    :return:
    """
    return guideline_path.read_text()