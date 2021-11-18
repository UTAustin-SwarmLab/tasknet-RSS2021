import torch
from torch import optim
from tqdm.auto import tqdm
import utils
#import visual
from torchvision.utils import save_image

def train_model(model, dataset, epochs=10,
                batch_size=32, sample_size=32,
                lr=3e-04, weight_decay=1e-5,
                loss_log_interval=30,
                image_log_interval=300,
                checkpoint_dir='./checkpoints',
                resume=False,
                cuda=False,
                results_dir = None,
                pixel_dim_x = None,
                num_channels = 3,
                zdim = None,
		device = None,
		dataset_name = None):
    print('cuda:', cuda)
    # prepare optimizer and model
    model.train()
    optimizer = optim.Adam(
        model.parameters(), lr=lr,
        weight_decay=weight_decay,
    )

    if resume:
        epoch_start = utils.load_checkpoint(model, checkpoint_dir)
    else:
        epoch_start = 1

    results_fname = results_dir + '/loss_results_' + '_'.join(['dataset', dataset_name, 'zdim', str(zdim)]) + '.txt' 
    
    results_f = open(results_fname, 'w')
    header_str = '\t'.join(['epoch', 'iteration', 'total_loss', 'recon_loss', 'kl_loss', 'dataset_name', 'zdim'])
    results_f.write(header_str + '\n')

    for epoch in range(epoch_start, epochs+1):

        data_loader = utils.get_data_loader(dataset, batch_size, cuda=cuda)
        data_stream = tqdm(enumerate(data_loader, 1))

        for batch_index, data_vec in data_stream:
            x = data_vec[0].to(device)

            # where are we?
            iteration = (epoch-1)*(len(dataset)//batch_size) + batch_index

            # flush gradients and run the model forward
            optimizer.zero_grad()
            (mean, logvar), x_reconstructed = model(x)
            reconstruction_loss = model.reconstruction_loss(x_reconstructed, x)
            kl_divergence_loss = model.kl_divergence_loss(mean, logvar)
            total_loss = reconstruction_loss + kl_divergence_loss

            # backprop gradients from the loss
            total_loss.backward()
            optimizer.step()

            # update progress in TQDM
            data_stream.set_description((
                'epoch: {epoch} | '
                'iteration: {iteration} | '
                'progress: [{trained}/{total}] ({progress:.0f}%) | '
                'loss => '
                'total: {total_loss:.4f} / '
                're: {reconstruction_loss:.3f} / '
                'kl: {kl_divergence_loss:.3f}'
            ).format(
                epoch=epoch,
                iteration=iteration,
                trained=batch_index * len(x),
                total=len(data_loader.dataset),
                progress=(100. * batch_index / len(data_loader)),
                total_loss=total_loss.data.item(),
                reconstruction_loss=reconstruction_loss.data.item(),
                kl_divergence_loss=kl_divergence_loss.data.item(),
            ))

            if iteration % loss_log_interval == 0:
                total_loss_str = '\t'.join([str(epoch), str(iteration), str(total_loss.data.item()), str(reconstruction_loss.data.item()), str(kl_divergence_loss.data.item()), str(dataset_name), str(zdim)])
                results_f.write(total_loss_str + '\n')
                results_f.flush()

            if iteration % image_log_interval == 0:
                data = x
                n = min(data.size(0), 16)
                comparison = torch.cat([data[:n], x_reconstructed.view(batch_size, num_channels, pixel_dim_x, pixel_dim_x)[:n]])
                save_image(comparison.cpu(), results_dir + '/zdim_' + str(zdim) + '_reconstruction_' + str(iteration) + '.png', nrow=n)

        # notify that we've reached to a new checkpoint.
        print()
        print('#############')
        print('# checkpoint!')
        print('#############')
        print()

        # save the checkpoint.
        utils.save_checkpoint(model, checkpoint_dir, epoch)
        print()
    results_f.close()
