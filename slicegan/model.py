from slicegan import preprocessing, util
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import time
import matplotlib
import os
from torch.utils.tensorboard import SummaryWriter, summary

def save_checkpoint(epoch, netG, netDs, optG, optDs, pth):
    torch.save({
        'epoch': epoch,
        'Generator_state_dict': netG.state_dict(),
        'Discriminator_state_dicts': [netD.state_dict() for netD in netDs],
        'optimizerG_state_dict': optG.state_dict(),
        'optimizerDs_state_dicts': [optD.state_dict() for optD in optDs]
    }, f"{pth}_checkpoint_{epoch}.pth")
    print(f"Checkpoint saved at epoch {epoch}")

def load_checkpoint(checkpoint_path, netG, netDs, optG, optDs, wass_dis_name, gp_name, loss_name, Wass_log, gp_log, disc_real_log, disc_fake_log, pth):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch']
        netG.load_state_dict(checkpoint['Generator_state_dict'])

        for i, netD in enumerate(netDs):
            netD.load_state_dict(checkpoint['Discriminator_state_dicts'][i])

        optG.load_state_dict(checkpoint['optimizerG_state_dict'])

        for i, optD in enumerate(optDs):
            optD.load_state_dict(checkpoint['optimizerDs_state_dicts'][i])

        print(f"Checkpoint loaded. Resuming training from epoch {start_epoch}.")
        with open(wass_dis_name, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                Wass_log.append(float(line))

        util.graph_plot([Wass_log], ['Wass Distance'], pth, 'WassGraph.jpg')
        with open(gp_name, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                gp_log.append(float(line))
        util.graph_plot([gp_log], ['Gradient Penalty'], pth, 'GpGraph.jpg')
        with open(loss_name, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                parts = line.split(',')
                disc_real_log.append(float(parts[0]))
                disc_fake_log.append(float(parts[1]))
        util.graph_plot([disc_real_log, disc_fake_log], ['real', 'perp'], pth, 'LossGraph.jpg')
        return start_epoch
    else:
        print("No checkpoint found. Starting training from scratch.")
        return 0  

def he_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')


def orthogonal_init(module):
    """
    Applies Orthogonal initialization to the weights of Conv2d and Linear layers.

    Args:
        module (nn.Module): A PyTorch module.
    """
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(module.weight)

def train(pth, Project_name, imtype, datatype, real_data, Disc, Gen, nc, l, nz, sf):
    """
    train the generator
    :param pth: path to save all files, imgs and data
    :param Project_name: name of the project
    :param imtype: image type e.g nphase, colour or gray
    :param datatype: training data format e.g. tif, jpg ect
    :param real_data: path to training data
    :param Disc:
    :param Gen:
    :param nc: channels
    :param l: image size
    :param nz: latent vector size
    :param sf: scale factor for training data 
    :return:
    """
    if len(real_data) == 1:
        real_data *= 3
        isotropic = True
    else:
        isotropic = False

    print('Loading Dataset...')
    dataset_xyz = preprocessing.batch(real_data, datatype, l, sf)

    ## Constants for NNs
    matplotlib.use('Agg')
    ngpu = 1
    # num_epochs = 100
    num_epochs = 100

    # batch sizes
    batch_size = 8
    D_batch_size = 8
    lrg = 0.0001
    # lrg = 0.001
    lrd = 0.0001
    # lrd = 0.001
    beta1 = 0.9
    beta2 = 0.99
    Lambda = 5
    critic_iters = 10
    cudnn.benchmark = True 
    workers = 0
    # 64:4, 128:6, 256:10
    lz = 4
    disc_real_log = []
    disc_fake_log = []
    gp_log = []
    Wass_log = []
    ##Dataloaders for each orientation
    device = torch.device("cuda:0" if(torch.cuda.is_available() and ngpu > 0) else "cpu")
    print(device, " will be used.\n")
    parent_path = os.path.dirname(pth)
    pth_ = pth
    pth = pth + '/' + Project_name
    use_tensorboard = False
    if use_tensorboard:
        slice_writer = SummaryWriter(log_dir=parent_path + '/tensorboard')
    pth_slice = parent_path + '/slices/'
    pth_checkpoint = parent_path + '/checkpoint/'
    wass_dis_name = pth_ + '/wass_dis.txt'
    gp_name = pth_ + '/gp.txt'
    loss_name = pth_ + '/loss.txt'
    if not os.path.exists(wass_dis_name):
        with open (wass_dis_name, 'w') as f:
            pass
        with open(gp_name, 'w') as f:
            pass
        with open(loss_name, 'w') as f:
            pass
    # D trained using different data for x, y and z directions
    dataloaderx = torch.utils.data.DataLoader(dataset_xyz[0], batch_size=batch_size,
                                              shuffle=True, num_workers=workers)
    dataloadery = torch.utils.data.DataLoader(dataset_xyz[1], batch_size=batch_size,
                                              shuffle=True, num_workers=workers)
    dataloaderz = torch.utils.data.DataLoader(dataset_xyz[2], batch_size=batch_size,
                                              shuffle=True, num_workers=workers)

    # Create the Genetator network
    netG = Gen().to(device)
    # netG.apply(he_init)
    if ('cuda' in str(device)) and (ngpu > 1): 
        netG = nn.DataParallel(netG, list(range(ngpu))) 
    optG = optim.Adam(netG.parameters(), lr=lrg, betas=(beta1, beta2)) 
    # optG = optim.RMSprop(netG.parameters(), lr=lrg)

    # Define 1 Discriminator and optimizer for each plane in each dimension
    netDs = []
    optDs = []
    for i in range(3):
        netD = Disc()
        # netD.apply(he_init)
        netD = (nn.DataParallel(netD, list(range(ngpu)))).to(device)
        netDs.append(netD)
        optDs.append(optim.Adam(netDs[i].parameters(), lr=lrd, betas=(beta1, beta2)))
        # optDs.append(optim.RMSprop(netDs[i].parameters(), lr=lrd))
    # ASPPFG = ASPPF.BasicRFB(nc, nc).to(device)

    init_img_D = torch.zeros((batch_size, nc, l, l), device=device)
    noise = torch.randn(D_batch_size, nz, lz, lz, lz, device=device)

    if use_tensorboard:
        with torch.no_grad():
            slice_writer.add_graph(netD, init_img_D, use_strict_trace=False)
            # fake_data = netG(noise)
            slice_writer.add_graph(netG, noise, use_strict_trace=False)

    print("Starting Training Loop...")
    # For each epoch
    start = time.time()

    start_epoch = load_checkpoint(pth_checkpoint + '_checkpoint_60.pth', netG,
                                  netDs, optG, optDs, wass_dis_name, gp_name, loss_name,
                                  Wass_log, gp_log, disc_real_log, disc_fake_log, pth)
    tags_E = ["Disc_Cost_E", "Gradient_Penalty_E", "Out_Real_E", "Out_Fake_E", "Wass_Distance_E", "lr_E"]

    for epoch in range(start_epoch, num_epochs):
        # sample data for each direction 
        for i, (datax, datay, dataz) in enumerate(zip(dataloaderx, dataloadery, dataloaderz), 1): # enumerate 用于获取索引及具体的值
            dataset = [datax, datay, dataz]
            ### Initialise
            ### Discriminator
            ## Generate fake image batch with G
            noise = torch.randn(D_batch_size, nz, lz, lz, lz, device=device) 
            fake_data = netG(noise).detach()
            # for each dim (d1, d2 and d3 are used as permutations to make 3D volume into a batch of 2D images)
            for dim, (netD, optimizer, data, d1, d2, d3) in enumerate(
                    zip(netDs, optDs, dataset, [2, 3, 4], [3, 2, 2], [4, 4, 3])):
                if isotropic: 
                    netD = netDs[0]
                    optimizer = optDs[0]
                netD.zero_grad() 
                ##train on real images
                real_data = data[0].to(device)
                real_data = real_data.requires_grad_(True)

                out_real = netD(real_data).view(-1).mean()
                fake_data_perm = fake_data.permute(0, d1, 1, d2, d3).reshape(l * D_batch_size, nc, l, l)
                # fake_data_perm_clone = fake_data_perm.clone()
                # fake_data_perm = ASPPFG(fake_data_perm)
                # out_fake_list = []
                out_fake = netD(fake_data_perm).mean()

                gradient_penalty = util.calc_gradient_penalty(netD, real_data, fake_data_perm[:batch_size],
                                                                      batch_size, l,
                                                                      device, Lambda, nc)
                disc_cost = out_fake - out_real + gradient_penalty 
                disc_cost.backward(retain_graph=True) 
                # nn.utils.clip_grad_norm_(netD.parameters(), max_norm=0.8)
                optimizer.step() 

            #logs for plotting
            disc_real_log.append(out_real.item())
            disc_fake_log.append(out_fake.item())
            Wass_log.append(out_real.item() - out_fake.item())
            gp_log.append(gradient_penalty.item())

            ### Generator Training
            if i % int(critic_iters) == 0:
                netG.zero_grad()
                errG = 0
                noise = torch.randn(batch_size, nz, lz,lz,lz, device=device)
                fake = netG(noise)

                for dim, (netD, d1, d2, d3) in enumerate(
                        zip(netDs, [2, 3, 4], [3, 2, 2], [4, 4, 3])):
                    if isotropic:
                        #only need one D
                        netD = netDs[0]
                    fake_data_perm = fake.permute(0, d1, 1, d2, d3).reshape(l * batch_size, nc, l, l)
                    output = netD(fake_data_perm)
                    errG -= output.mean()
                    # Calculate gradients for G
                errG.backward()
                optG.step()

            # Output training stats & show imgs
            if i % 25 == 0:
                netG.eval()
                with torch.no_grad():
                    torch.save(netG.state_dict(), pth + '_Gen.pt')
                    torch.save(netD.state_dict(), pth + '_Disc.pt')
                    noise = torch.randn(1, nz,lz,lz,lz, device=device)
                    img = netG(noise)
                    ###Print progress
                    ## calc ETA
                    steps = len(dataloaderx)
                    util.calc_eta(steps, time.time(), start, i, epoch, num_epochs)
                    ###save example slices
                    util.test_plotter(img, 5, imtype, pth_slice, epoch)
                    # plotting graphs
                    util.graph_plot([disc_real_log, disc_fake_log], ['real', 'perp'], pth, 'LossGraph.jpg')
                    util.graph_plot([Wass_log], ['Wass Distance'], pth, 'WassGraph.jpg')
                    util.graph_plot([gp_log], ['Gradient Penalty'], pth, 'GpGraph.jpg')
                    with open(wass_dis_name, 'a') as file:
                        for i in range(len(Wass_log)):
                            file.write(f"{Wass_log[i]}\n")
                    with open(gp_name, 'a') as file:
                        for i in range(len(gp_log)):
                            file.write(f"{gp_log[i]}\n")
                    with open(loss_name, 'a') as file:
                        for i in range(len(disc_real_log)):
                            file.write(f"{disc_real_log[i]},{disc_fake_log[i]}\n")
                netG.train()

        if (epoch + 1) % 5 == 0:  
            save_checkpoint(epoch + 1, netG, netDs, optG, optDs, pth_checkpoint)

        if use_tensorboard:
            for name, param in netD.named_parameters():
                # print(f"Layer: {name}, Shape: {param.shape}")
                # print(param.data)
                if "weight" in name:
                    slice_writer.add_histogram(f"layer_{name}_weight", param.clone().cpu().data.numpy(), global_step=epoch)
            slice_writer.add_scalar(tags_E[0], disc_cost, epoch)
            slice_writer.add_scalar(tags_E[1], gradient_penalty, epoch)
            slice_writer.add_scalar(tags_E[2], out_real, epoch)
            slice_writer.add_scalar(tags_E[3], out_fake, epoch)
            slice_writer.add_scalar(tags_E[4], out_real.item() - out_fake.item(), epoch)
            slice_writer.add_scalar(tags_E[5], optimizer.param_groups[0]['lr'], epoch)


    if use_tensorboard:
        slice_writer.close()
