class Corpus(object):
    def __init__(self, path, device):
        self.dictionary = Dictionary()
        self.isPennTreeBank = "penntreebank" in path
        self.train = self.tokenize(os.path.join(path, 'train.txt'), isTrain=True).to(device)
        self.valid = self.tokenize(os.path.join(path, 'valid.txt')).to(device)
        self.test = self.tokenize(os.path.join(path, 'test.txt')).to(device)

    def tokenize(self, path, isTrain=False):
        assert os.path.exists(path)

        if isTrain:
            with open(path, 'r', encoding="utf8") as file:
                for line in file:
                    if self.isPennTreeBank:
                        words = ['<sos>'] + line.split() + ['<eos>']
                    else:
                        # replace all numbers with 'N'
                        line = re.sub(r'[0-9]+','N',line)
                        line = line.replace("=", '').replace('.', '<eos> <sos>')
                        words = line.split()
                    for word in words:
                        self.dictionary.add_word(word)

        with open(path, 'r', encoding="utf8") as f:
            data = []
            for line in f:
                if self.isPennTreeBank:
                    words = ['<sos>'] + line.split() + ['<eos>']
                else:
                    # replace all numbers with 'N'
                    line = re.sub(r'[0-9]+', 'N', line)
                    line = line.replace("=", '').replace('.','<eos> <sos>')
                    words = line.split()
                sent_tokenized = torch.tensor([self.dictionary.word2idx[word] for word in words]).type(torch.int64)
                data.append(sent_tokenized)
            return torch.cat(data)