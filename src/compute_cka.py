from src.utils.model_loader import load_model
import argparse
from utils.cka_utils import get_cka_layers, save_cka_activations, save_cka_matrix

parser = argparse.ArgumentParser(description='CKA similarity compute')

parser.add_argument("--model1", type=str, default="clip-res",
                    choices=['clip-res', 'cvcl-resnext', 'resnext'],
                    help="First(Y axis in matrix) model in the pair for comparison")
parser.add_argument("--model2", type=str, default="cvcl-resnext", #! this is basically fixed as x-axis to be cvcl for comparison in final plot
                    choices=['clip-res', 'cvcl-resnext', 'resnext'],
                    help="Second(X axis in matrix) model in the pair for comparison")
parser.add_argument("--layers", type=str, default="layer1,layer2,layer3,layer4",
                    help="Layer names separated by commas (no spaces)")
parser.add_argument("--d_probe", type=str, default="objects", 
                    choices = ["objects", "imagenet_val", "broden"])
parser.add_argument("--batch_size", type=int, default=128, help="Batch size when running CLIP/target model")
parser.add_argument("--device", type=str, default="cuda", help="whether to use GPU/which gpu")
parser.add_argument("--activation_dir", type=str, default="./experiments/CKA/test_activations", help="where to save activations")
parser.add_argument("--result_dir", type=str, default="./experiments/CKA/test_similarity_matrix", help="where to save calculated CKA matrix")
parser.add_argument("--pool_mode", type=str, default="avg", help="Aggregation function for channels, max or avg")

parser.parse_args()


if __name__ == '__main__':
    args = parser.parse_args()

    device = args.device

    model1, transform1 = load_model(model_name=args.model1, device=device)
    model2, transform2 = load_model(model_name=args.model2, device=device)

    layers = args.layers.split(',')
    model_layers_1 = get_cka_layers(model1, layers)
    model_layers_2 = get_cka_layers(model2, layers)  

    d_probe = args.d_probe
    batch_size = args.batch_size
    activation_dir = args.activation_dir
    pool_mode = args.pool_mode

    save_cka_activations(args.model1, model1, transform1, model_layers_1, d_probe, batch_size, device, pool_mode, activation_dir)
    save_cka_activations(args.model2, model2, transform2, model_layers_2, d_probe, batch_size, device, pool_mode, activation_dir)
    
    save_cka_matrix(d_probe, activation_dir, model_layers_1, model_layers_2, args)
    