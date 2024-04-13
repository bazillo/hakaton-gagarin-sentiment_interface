MAX_LENGTH = 256
BATCH_SIZE = 32
LR = 3e-4
NUM_EPOCHS = 3
FREEZE_BACKBONE = False

def wrap(text, aspect):
    """Функция которая оборачивает текст и аспект в формат, принимаемый токенайзером"""
    return f"[CLS] {text} [SEP] {aspect} [SEP]"