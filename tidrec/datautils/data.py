from tidrec.datautils.data_load import *
from tidrec.datautils.dataset import *
from tidrec.utils import *
from torch.utils.data import DataLoader


class Data:
    def __init__(self, conf):
        self.conf = conf
        if conf.data_name == 'dblp' or conf.data_name == 'wos':
            self.all_author_pair_set, self.all_ui_pair_dict = load_data_v1(self.conf)
        else:
            raise NotImplementedError('Wrong data name')
        self.all_ui_pair_set = get_pair_set_from_pair_dict(self.all_ui_pair_dict)

        self.author_set = set()
        self.paper_set = set()
        for author, paper in self.all_ui_pair_set:
            self.author_set.add(author)
            self.paper_set.add(paper)
        print('Num authors', len(self.author_set))
        print('Num papers', len(self.paper_set))

        self.train_ui_pair_dict, self.valid_ui_pair_dict, self.test_ui_pair_dict = \
            split_pair_dict_random_ratio(self.all_ui_pair_set, 0.15, 0.05, conf.train_ratio)
        # author
        self.train_author_set = set(self.train_ui_pair_dict.keys())
        self.valid_author_set = set(self.valid_ui_pair_dict.keys())
        self.test_author_set = set(self.test_ui_pair_dict.keys())
        # paper
        self.train_ui_pair_set = get_pair_set_from_pair_dict(self.train_ui_pair_dict)
        self.valid_ui_pair_set = get_pair_set_from_pair_dict(self.valid_ui_pair_dict)
        self.test_ui_pair_set = get_pair_set_from_pair_dict(self.test_ui_pair_dict)
        self.train_iu_pair_set = reverse_pair_set(self.train_ui_pair_set)  # 反转训练集

        print('Num authors: train:{}, valid:{}, test:{}'.format(len(self.train_ui_pair_dict),
                                                              len(self.valid_ui_pair_dict),
                                                              len(self.test_ui_pair_dict)))
        print('Num ui pairs: train:{}, valid:{}, test:{}'.format(len(self.train_ui_pair_set),
                                                                 len(self.valid_ui_pair_set),
                                                                 len(self.test_ui_pair_set)))

        # ('author','co-write','author'), ('author','write','paper')
        self.data_graph = construct_hetero_graphs(
            [self.all_author_pair_set, self.train_ui_pair_set, self.train_iu_pair_set],
            [('author', 'friend', 'author'), ('author', 'like', 'paper'), ('paper', 'rev_like', 'author')],
            {'author': conf.num_authors, 'paper': conf.num_papers}
        ).to(self.conf.device)

        self.train_ui_pair2eid = self.get_pair2eid(self.train_ui_pair_set)
        self.train_iu_pair2eid = self.get_pair2eid(self.train_iu_pair_set)

        # train
        self.train_dataset = Train_dataset(conf, self.train_ui_pair_set)
        self.train_data_loader = DataLoader(self.train_dataset, batch_size=conf.train_batch_size,
                                            collate_fn=self.collate_train, shuffle=True)

        # eval positive
        self.train_eval_author_idx_dict, self.train_eval_author_list, self.train_eval_paper_list = \
            self.get_eval_positive_dataset(self.train_ui_pair_set)
        self.valid_eval_author_idx_dict, self.valid_eval_author_list, self.valid_eval_paper_list = \
            self.get_eval_positive_dataset(self.valid_ui_pair_set)
        self.test_eval_author_idx_dict, self.test_eval_author_list, self.test_eval_paper_list = \
            self.get_eval_positive_dataset(self.test_ui_pair_set)

        # eval negative
        self.train_data_dict = Eval_dataset(conf, self.train_author_set, self.train_ui_pair_dict)
        self.valid_data_dict = Eval_dataset(conf, self.valid_author_set, self.valid_ui_pair_dict)
        self.test_data_dict = Eval_dataset(conf, self.test_author_set, self.test_ui_pair_dict)
        self.train_neg_data_loader_train = DataLoader(self.train_data_dict, batch_size=conf.train_batch_size,
                                                      collate_fn=self.collate_eval_neg_train, shuffle=False)
        self.valid_neg_data_loader_train = DataLoader(self.valid_data_dict, batch_size=conf.train_batch_size,
                                                      collate_fn=self.collate_eval_neg_train, shuffle=False)
        self.test_neg_data_loader_train = DataLoader(self.test_data_dict, batch_size=conf.train_batch_size,
                                                     collate_fn=self.collate_eval_neg_train, shuffle=False)

        self.train_neg_data_loader_test = DataLoader(self.train_data_dict, batch_size=conf.test_batch_size,
                                                     collate_fn=self.collate_eval_neg_test, shuffle=False)
        self.valid_neg_data_loader_test = DataLoader(self.valid_data_dict, batch_size=conf.test_batch_size,
                                                     collate_fn=self.collate_eval_neg_test, shuffle=False)
        self.test_neg_data_loader_test = DataLoader(self.test_data_dict, batch_size=conf.test_batch_size,
                                                    collate_fn=self.collate_eval_neg_test, shuffle=False)
        print('Finish data initialize')

    @staticmethod
    def read_co_data(co_data):
        author_pair_set = set()
        for row in range(co_data.shape[0]):
            u1_id, u2_id = int(co_data[row][0]) - 1, int(co_data[row][1]) - 1
            author_pair_set.add((u1_id, u2_id))
        return author_pair_set

    @staticmethod
    def read_write_data(write_data):
        ui_pair_dict = defaultdict(set)
        for row in range(write_data.shape[0]):
            write_paper = write_data[row]
            author_id, paper_id = int(write_paper[0]) - 1, int(write_paper[1]) - 1,
            category, score = int(write_paper[2]), int(write_paper[3])
            ui_pair_dict[author_id].add(paper_id)
        return ui_pair_dict

    # generate train data
    def collate_train(self, samples):
        author_list, paper_list = map(list, zip(*samples))
        collected_author_list = []
        collected_paper_list = []
        collected_label_list = []
        for idx in range(len(author_list)):
            collected_author_list.append(author_list[idx])
            collected_paper_list.append(paper_list[idx])
            collected_label_list.append(1)
            collected_author_list.extend([author_list[idx]] * self.conf.num_train_negatives)
            for _ in range(self.conf.num_train_negatives):
                negative_paper_id = np.random.randint(self.conf.num_papers)
                while (author_list[idx], negative_paper_id) in self.train_ui_pair_set:
                    negative_paper_id = np.random.randint(self.conf.num_papers)
                collected_paper_list.append(negative_paper_id)
            collected_label_list.extend([0] * self.conf.num_train_negatives)

        return collected_author_list, collected_paper_list, collected_label_list

    # generate eval data
    @staticmethod
    def get_eval_positive_dataset(ui_pair_set):
        author_list = []
        paper_list = []
        author_idx_dict = defaultdict(list)
        for idx, (author_id, paper_id) in enumerate(ui_pair_set):
            author_list.append(author_id)
            paper_list.append(paper_id)
            author_idx_dict[author_id].append(idx)
        return author_idx_dict, author_list, paper_list

    def collate_eval_neg_train(self, samples):
        author_list, paper_list = map(list, zip(*samples))
        collected_author_list = []
        collected_paper_list = []
        for idx in range(len(author_list)):
            collected_author_list.extend([author_list[idx]] * self.conf.num_eval_negatives)
            for _ in range(self.conf.num_eval_negatives):
                negative_paper_id = np.random.randint(self.conf.num_papers)
                while (author_list[idx], negative_paper_id) in self.all_ui_pair_set:
                    negative_paper_id = np.random.randint(self.conf.num_papers)
                collected_paper_list.append(negative_paper_id)
        return author_list, collected_author_list, collected_paper_list

    def collate_eval_neg_test(self, samples):
        author_list, paper_list = map(list, zip(*samples))
        collected_author_list = []
        collected_paper_list = []
        for idx in range(len(author_list)):
            collected_author_list.extend([author_list[idx]] * self.conf.num_test_negatives)
            for _ in range(self.conf.num_test_negatives):
                negative_paper_id = np.random.randint(self.conf.num_papers)
                while (author_list[idx], negative_paper_id) in self.all_ui_pair_set:
                    negative_paper_id = np.random.randint(self.conf.num_papers)
                collected_paper_list.append(negative_paper_id)

        return author_list, collected_author_list, collected_paper_list

    @staticmethod
    def get_pair2eid(pair_list):
        pair2eid = {}
        for eid, pair in enumerate(pair_list):
            pair2eid[pair] = eid
        return pair2eid
