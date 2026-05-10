import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from jarvis.assistant import JarvisAssistant


def main() -> None:
    assistant = JarvisAssistant()
    assistant.run()


if __name__ == "__main__":
    main()