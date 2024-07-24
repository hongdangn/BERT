from tokenizers import BertWordPieceTokenizer
import os

def make_vocab(data_path: str):
    # data_path: path to data files after preprocesss
    paths = []
    for file in os.listdir(data_path):
        paths.append(os.path.join(data_path, file))

    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=False,
        strip_accents=False,
        lowercase=True
    )

    tokenizer.train(
        files=paths,
        vocab_size=30_000,
        min_frequency=5,
        limit_alphabet=1000,
        wordpieces_prefix='##',
        special_tokens=['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]']
        )

    os.mkdir('./tokenizer')
    tokenizer.save_model('./tokenizer', 'bert')