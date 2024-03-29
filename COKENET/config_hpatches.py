import argparse
import numpy as np

def get_config():
    parser = argparse.ArgumentParser(description='HSequences Extract Features')
    parser.add_argument('--data-dir', type=str, default='/data0/ZHL/dataset/',
                        help='The root path to HSequences dataset.')
    ## basic configuration
    parser.add_argument('--list-images', type=str, default='test_im/image.txt',
                        help='File containing the image paths for extracting features.')
    parser.add_argument('--results-dir', type=str, default='/data0/ZHL/project/keynet/COKE/extracted_features/',
                        help='The output path to save the extracted keypoint.')
    parser.add_argument('--network-version', type=str, default='coke_20_s_2_8',
                        help='The COKE version name')
    parser.add_argument('--checkpoint-det-dir', type=str, default='COKENET/pretrained_nets/ab/bs_2_8.pt',
                        help='The path to the checkpoint file to load the detector weights.')  #bs_final_1_1.pt
    parser.add_argument('--checkpoint-coke-dir', type=str, default='COKENET/pretrained_nets/ab/sp_2_8.pt',
                        help='The path to the checkpoint file to load the detector weights.')  #sp_final_1_1.pt
    parser.add_argument('--pytorch-hardnet-dir', type=str, default='COKENET/pretrained_nets/HardNet++.pth',
                        help='The path to the checkpoint file to load the HardNet descriptor weights.')
    # Detector Settings
    parser.add_argument('--batch-size', type=int, default=1,
                        help='The batch size for training.')
    parser.add_argument('--patch-size', type=int, default=192,
                        help='The patch size of the generated dataset.')                        
    parser.add_argument('--num-filters', type=int, default=8,
                        help='The number of filters in each learnable block.')
    parser.add_argument('--num-learnable-blocks', type=int, default=3,
                        help='The number of learnable blocks after handcrafted block.')
    parser.add_argument('--num-levels-within-net', type=int, default=3,
                        help='The number of pyramid levels inside the architecture.')
    parser.add_argument('--factor-scaling-pyramid', type=float, default=1.2,
                        help='The scale factor between the multi-scale pyramid levels in the architecture.')
    parser.add_argument('--conv-kernel-size', type=int, default=5,
                        help='The size of the convolutional filters in each of the learnable blocks.')
  # Multi-Scale Extractor Settings
    parser.add_argument('--extract-MS', type=bool, default=False,
                        help='Set to True if you want to extract multi-scale features.')
    parser.add_argument('--num-points', type=int, default=1000,  #1500
                        help='The number of desired features to extract.')
    parser.add_argument('--nms-size', type=int, default=15,  #15
                        help='The NMS size for computing the validation repeatability.')
    parser.add_argument('--border-size', type=int, default=15,
                        help='The number of pixels to remove from the borders to compute the repeatability.')
    parser.add_argument('--order-coord', type=str, default='xysr',
                        help='The coordinate order that follows the extracted points. Use yxsr or xysr.')
    parser.add_argument('--random-seed', type=int, default=12345,
                        help='The random seed value for TensorFlow and Numpy.')
    parser.add_argument('--pyramid_levels', type=int, default=5,  #5
                        help='The number of downsample levels in the pyramid.')
    parser.add_argument('--upsampled-levels', type=int, default=1,  #1
                        help='The number of upsample levels in the pyramid.')
    parser.add_argument('--scale-factor-levels', type=float, default=np.sqrt(2),
                        help='The scale factor between the pyramid levels.')
    parser.add_argument('--scale-factor', type=float, default=2.,
                        help='The scale factor to extract patches before descriptor.')
    args = parser.parse_args()

    return args


def get_eval_config():

    parser = argparse.ArgumentParser(description='HSequences Compute Repeatability')

    parser.add_argument('--data-dir', type=str, default='/data0/ZHL/dataset/hpatches-sequences-release',
                        help='The root path to HSequences dataset.')
    parser.add_argument('--results-bench-dir', type=str, default='HSequences_bench/results/',
                        help='The output path to save the results.')
    parser.add_argument('--detector-name', type=str, default='coke_20_s_2_8',
                        help='The name of the detector to compute metrics.')
    parser.add_argument('--results-dir', type=str, default='/data0/ZHL/project/keynet/COKE/extracted_features/',
                        help='The path to the extracted points.')
    parser.add_argument('--split', type=str, default='view',
                        help='The name of the HPatches (HSequences) split. Use full, debug_view, debug_illum, view or illum.')
    parser.add_argument('--split-path', type=str, default='HSequences_bench/splits.json',
                        help='The path to the split json file.')
    parser.add_argument('--top-k-points', type=int, default=1000, #1000
                        help='The number of top points to use for evaluation. Set to None to use all points')
    parser.add_argument('--overlap', type=float, default=0.6,
                        help='The overlap threshold for a correspondence to be considered correct.')
    parser.add_argument('--pixel-threshold', type=int, default=5,
                        help='The distance of pixels for a matching correspondence to be considered correct.')

    parser.add_argument('--dst-to-src-evaluation', type=bool, default=True,
                        help='Order to apply homography to points. Use True for dst to src, False otherwise.')
    parser.add_argument('--order-coord', type=str, default='xysr',
                        help='The coordinate order that follows the extracted points. Use either xysr or yxsr.')

    args = parser.parse_args()

    return args
