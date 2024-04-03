from .scannet import ScannetDetectionDataset, ScannetDatasetConfig
# from .sunrgbd import SunrgbdDetectionDataset, SunrgbdDatasetConfig #will release very soon


DATASET_FUNCTIONS = {
    "scannet": [ScannetDetectionDataset, ScannetDatasetConfig],}


def build_dataset(args):
    dataset_builder = DATASET_FUNCTIONS[args.dataset_name][0]
    dataset_config = DATASET_FUNCTIONS[args.dataset_name][1]()
    
    if args.test_only:
        dataset_dict = {
            "test": dataset_builder(
                dataset_config, 
                split_set="val", 
                augment=False,
                args=args
            ),
        }
    else:
        dataset_dict = {
            "train": dataset_builder(
                dataset_config, 
                split_set="train", 
                augment=True,
                args=args
            ),
            "test": dataset_builder(
                dataset_config, 
                split_set="val", 
                augment=False,
                args=args
            ),
        }
    return dataset_dict, dataset_config
    