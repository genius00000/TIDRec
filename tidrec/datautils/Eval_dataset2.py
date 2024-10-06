from torch.utils.data import Dataset


class Eval_dataset2(Dataset):
    def __init__(self, author_set, ui_pair_dict):
        super(Eval_dataset2, self).__init__()
        # self.conf = conf
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