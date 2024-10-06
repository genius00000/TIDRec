from torch.utils.data import Dataset


class Eval_dataset(Dataset):
    def __init__(self, conf, author_set, ui_pair_dict):
        super(Eval_dataset, self).__init__()
        self.conf = conf
        self.author_list = list(author_set)
        self.paper_list = self.get_paper_label_list(ui_pair_dict)

    def get_paper_label_list(self, ui_pair_dict):
        paper_list = []
        for author in self.author_list:
            author_paper_list = list(ui_pair_dict[author])
            paper_list.append(author_paper_list)
        return paper_list

    def __getpaper__(self, idx):
        return self.author_list[idx], self.paper_list[idx]

    def __len__(self):
        return len(self.author_list)


class Train_dataset(Dataset):
    def __init__(self, conf, ui_pair_set):
        super(Train_dataset, self).__init__()
        self.conf = conf
        self.ui_pair_list = list(ui_pair_set)

    def __getpaper__(self, idx):
        return self.ui_pair_list[idx]

    def __len__(self):
        return len(self.ui_pair_list)
