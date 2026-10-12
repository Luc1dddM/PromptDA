"""Shared dataset selection for SACG training and comparison inference."""
from dataset.scannetpp import ScanNetPPDataset

ARKIT_ROOT = '/media/naeem/T7/ArKitScene/ARKitScenes-main/data/upsampling'
SCANNETPP_ROOT = '/media/naeem/4TB/datasets/scannetpp_ours'


def add_dataset_args(parser):
    parser.add_argument('--dataset', choices=['arkitscenes', 'scannetpp'], default='arkitscenes')
    parser.add_argument('--data_seed', type=int, default=26, help='Fixed subset seed, independent of model/training seed')
    parser.add_argument('--data_root', default=None, help='Defaults to the selected dataset location')
    parser.add_argument('--image_height', type=int, default=756, help='ScanNet++ RGB/target height (multiple of 14)')
    parser.add_argument('--image_width', type=int, default=1008, help='ScanNet++ RGB/target width (multiple of 14)')
    parser.add_argument('--depth_scale', type=float, default=1000.0, help='ScanNet++ stored depth units per metre')
    parser.add_argument('--ignore_rgb_masks', action='store_true', help='Disable ScanNet++ RGB validity masks')


def resolve_dataset_args(args):
    if any(x <= 0 or x % 14 for x in (args.image_height, args.image_width)):
        raise ValueError("Image dimensions must be positive multiples of 14")
    if args.data_root is None:
        args.data_root = SCANNETPP_ROOT if args.dataset == 'scannetpp' else ARKIT_ROOT
    return args


def build_dataset(args, split, max_samples, transform):
    common = dict(root=args.data_root, split=split, max_samples=max_samples,
                  seed=args.data_seed, transform=transform)
    if args.dataset == 'scannetpp':
        return ScanNetPPDataset(**common, target_hw=(args.image_height, args.image_width),
                                depth_scale=args.depth_scale, use_rgb_masks=not args.ignore_rgb_masks)
    from dataset.dataset import MyARKitScenesDataset
    return MyARKitScenesDataset(**common, target_hw=(args.image_height, args.image_width))
