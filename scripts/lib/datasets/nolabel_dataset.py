import glob

from .lane_dataset_loader import LaneDatasetLoader


class NoLabelDataset(LaneDatasetLoader):
    def __init__(self, img_h=720, img_w=1280,dataset_type=None, max_lanes=None, root=None, img_ext='.jpg', **_):
        """Use this loader if you want to test a model on an image without annotations or implemented loader."""
        self.root = root
        if root is None:
            raise Exception('Please specify the root directory')

        self.img_w, self.img_h = img_w, img_h
        self.img_ext = img_ext
        self.dataset_type = dataset_type
        self.annotations = []
        self.load_annotations()

        print("--------------------")
        print(self.dataset_type)
        self.split="test"

        # Force max_lanes, used when evaluating testing with models trained on other datasets
        # On NoLabelDataset, always force it
        self.max_lanes = max_lanes

    def get_img_heigth(self, _):
        return self.img_h

    def get_img_width(self, _):
        return self.img_w

    def get_metrics(self, lanes, _):
        return 0, 0, [1] * len(lanes), [1] * len(lanes)

    def load_annotations(self):
        #print(self.max_lanes)
        #dataset=get_dataset()
        self.annotations = []
        if self.dataset_type=="kodasv3":
            tmp = ["2019Y07M05D15H43m44s", "2019Y07M05D15H48m42s", "2019Y07M05D16H01m58s", "2019Y07M05D16H18m12s",
                   "2019Y07M05D16H41m35s", "2019Y07M05D16H47m20s", "2019Y07M05D16H49m20s"]
            temp = []
            for i in tmp:
                pattern = '{}/'.format(self.root)
                print(pattern + i)
                pattern = pattern + i + ('/Camera_FrontMid/FrontMid/*{}'.format(self.img_ext))
                print('Looking for image files with the pattern', pattern)
                temp.append(sorted(glob.glob(pattern, recursive=True)))
            for i in temp:
                for file in i:
                    self.annotations.append({'lanes': [], 'path': file})
        elif self.dataset_type=="kodasv1":
            pattern = '{}/*{}'.format(self.root, self.img_ext)
            print('Looking for image files with the pattern', pattern)
            temp=sorted(glob.glob(pattern, recursive=True))
            for file in temp:
                self.annotations.append({'lanes': [], 'path': file})

    def eval(self, _, __, ___, ____, _____):
        return "", None

    def __getitem__(self, idx):
        idx = 0 # hansaem webcam
        return self.annotations[idx]

    def __len__(self):
        return len(self.annotations)
