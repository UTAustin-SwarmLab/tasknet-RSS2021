import os,argparse
import main
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class CPUModel:
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print('CPU model {} loaded'.format(model_path))

    def run_inference(self, input_tensor):
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data

def run(args):
    # load edge model
    if args.use_edgetpu:
        from edgetpu.basic import basic_engine
        edge_model = basic_engine.BasicEngine(args.edge_model)
    else:
        edge_model = CPUModel(args.edge_model)
    # load cloud model
    cloud_model = tf.keras.models.load_model(args.cloud_model)
    # load dataset
    test_dir = os.path.join(args.data_path, 'val') if os.path.exists(os.path.join(args.data_path, 'val')) else args.data_path
    data_generator = main.get_dir_generator(test_dir, (args.input_size,args.input_size), train=False, batch_size=1)
    classes = os.listdir(test_dir)
    classes.sort()
    # plot
    row, col = 2, 4
    fig, axs = plt.subplots(row, col)
    axs = axs.flatten()
    # test
    confidence, true_label, latency_stat = [], [], []
    for num_batch, data in enumerate(data_generator):
        if num_batch%100==0: print('batch {} started'.format(num_batch))
        x, y = data[0], data[1]
        if args.use_edgetpu:
            latency, edge_output = edge_model.run_inference(tf.reshape(tf.dtypes.cast(x, tf.uint8), [-1]))
            latency_stat.append(latency)
            edge_output = tf.reshape(edge_output, (-1, args.latent_dim))
        else:
            #edge_output = edge_model.run_inference(tf.dtypes.cast(x, tf.uint8))
            edge_output = edge_model.run_inference(x)
        #print(edge_output)
        output = cloud_model(edge_output)
        confidence.extend(output.numpy())
        true_label.extend(y)
        if num_batch < row*col:
            axs[num_batch].set_title('Prediction: {}\n True Label: {}'.format(classes[np.argmax(output.numpy())], classes[np.argmax(y)]), size=8)
            axs[num_batch].imshow(x[0].astype(int))
            axs[num_batch].axis("off")
        if num_batch == len(data_generator)-1: break
    if args.use_edgetpu: print('TPU latency (ms/img): {:.2f}'.format(np.mean(latency_stat)))
    cce = tf.keras.losses.CategoricalCrossentropy()
    loss = cce(true_label, confidence).numpy()
    true_labels = np.argmax(true_label, axis=1)
    preds = np.argmax(confidence, axis=1)
    acc = np.count_nonzero(true_labels==preds)/len(true_labels)
    print('loss: {:.5f}, accuracy: {:.5f}'.format(loss, acc))

    fig.tight_layout()
    plt.show()
    plt.close()

########### MAIN ###########
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--edge_model', type=str, help='path to edge model')
    parser.add_argument('--cloud_model', type=str, help='path to cloud model')
    parser.add_argument('--data_path', type=str, help='path to training and validation data')
    parser.add_argument('--input_size', type=int, default=224, help='input image size')
    parser.add_argument('--latent_dim', type=int, default=8, help='latent dimension')
    parser.add_argument('--use_edgetpu', action='store_true', help='load edge model on coral edgetpu')

    args = parser.parse_args()

    run(args)
