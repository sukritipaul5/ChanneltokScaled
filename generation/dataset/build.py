from .imagenet_npz import build_imagenet_npz


def build_dataset(args, **kwargs):
    if args.dataset == 'imagenet_npz':
        return build_imagenet_npz(args, **kwargs)

    raise ValueError(f'dataset {args.dataset} is not supported')
