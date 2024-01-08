if __name__ == '__main__':
# Load necessary packages
    import ssms
    import lanfactory 
    import os
    import numpy as np
    from copy import deepcopy
    import torch
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # os.chdir('D:horiz/IMPORTANT/0study_graduate/Pro_COMPASS/COMPASS_DDM/')
    #%% Generate training data
    # MAKE CONFIGS

    # Initialize the generator config (for MLP LANs)
    # generator_config = deepcopy(ssms.config.data_generator_config['lan']['mlp'])


    generator_config = deepcopy(ssms.config.data_generator_config['lan'])

    # Specify generative model (one from the list of included models mentioned above)
    ddm_model = "angle"
    generator_config['dgp_list'] = ddm_model
    # Specify number of parameter sets to simulate
    generator_config['n_parameter_sets'] = 100  # suggest not change
    # Specify how many samples a simulation run should entail
    generator_config['n_samples'] = 100
    # Specify folder in which to save generated data
    generator_config['output_folder'] = 'train_LAN/{}/data/'.format(ddm_model)
    # save config path
    save_configs_path = 'train_LAN/{}/config/'.format(ddm_model)
    # 'data/lan_mlp/sets'+str(generator_config['n_parameter_sets'])+'samples'+str(generator_config['n_samples'])+"/"
    print("generate data saving at: "+generator_config['output_folder'] )

    # Make model config dict
    model_config = ssms.config.model_config[ddm_model]


    #%% MAKE DATA
    my_dataset_generator = ssms.dataset_generators.data_generator(generator_config = generator_config,
                                                                model_config = model_config)
    # save data
    training_data = my_dataset_generator.generate_data_training_uniform(save = True)
    # 2 Prepare for training
    # LAN uses custom dataloaders. We expect to receive a bunch of training data so 
    # We need dataloader to spits out batches of data from a specific training data file, and keep checking when to load a new file

    # List of datafiles (here only one)
    # folder_ = 'data/lan_mlp/traindata_angle/'
    folder_ = generator_config['output_folder'] 
    file_list_ = [folder_ + file_ for file_ in os.listdir(folder_) if 'training_data' in file_]
    # network_file_path = [generator_config['output_folder'] + file_ for file_ in network_path_list if 'state_dict' in file_][0]
    # Training dataset
    torch_training_dataset = lanfactory.trainers.DatasetTorch(file_ids = file_list_,
                                                            batch_size = 128)

    torch_training_dataloader = torch.utils.data.DataLoader(torch_training_dataset,
                                                            shuffle = True,
                                                            batch_size = None,
                                                            num_workers = 1,
                                                            pin_memory = True)

    # Validation dataset
    torch_validation_dataset = lanfactory.trainers.DatasetTorch(file_ids = file_list_,
                                                            batch_size = 128)

    torch_validation_dataloader = torch.utils.data.DataLoader(torch_validation_dataset,
                                                            shuffle = True,
                                                            batch_size = None,
                                                            num_workers = 1,
                                                            pin_memory = True)
    
    network_config = lanfactory.config.network_configs.network_config_mlp

    print('Network config: ')
    print(network_config)

    train_config = lanfactory.config.network_configs.train_config_mlp
    
    train_config["n_epochs"] = 10

    print('Train config: ')
    print(train_config)
#%% 3
    ##################################
    ##################################
    ##################################
    # LOAD NETWORK
    net = lanfactory.trainers.TorchMLP(network_config = deepcopy(network_config),
                                    input_shape = torch_training_dataset.input_dim)
    #                                   save_folder = '/data/torch_models/',
    #                                   generative_model_id = 'angle')

    # SAVE CONFIGS
    lanfactory.utils.save_configs(model_id =ddm_model,
                                save_folder = save_configs_path, 
                                network_config = network_config, 
                                train_config = train_config, 
                                allow_abs_path_folder_generation = True)
    # TRAIN MODEL # need to change name?
    model_trainer=lanfactory.trainers.ModelTrainerTorchMLP(train_config=train_config,
                                            train_dl=torch_training_dataloader,
                                            valid_dl=torch_validation_dataloader,
                                            model=net)
    model_trainer.train_and_evaluate(save_history = True,
                            save_model = True,
                            output_folder= generator_config['output_folder'],
                            verbose = 0)
    #%% 8
    # network_path_list = os.listdir(generator_config['output_folder']+"/")
    # network_file_path = [generator_config['output_folder'] + file_ for file_ in network_path_list if 'state_dict' in file_][0]

    # network = lanfactory.trainers.LoadTorchMLPInfer(model_file_path = network_file_path,
    #                                                 network_config = network_config,
    #                                                 input_dim = torch_training_dataset.input_dim)

    # #%% 9
    # # Two ways to call the network
    # # Direct call --> need tensor input
    # # direct_out = network(torch.from_numpy(np.array([1, .5, 0.5, 1.0, 0.1, 0.65, 1], dtype  = np.float32)))
    # direct_out = network(torch.from_numpy(np.array([1, 0.5, 0.5, 0.5, 0.65, 1], dtype  = np.float32)))
    # print('direct call out: ', direct_out)

    # # predict_on_batch method
    # # predict_on_batch_out = network.predict_on_batch(np.array([1, 1.5, 0.5, 1.0, 0.1, 0.65, 1], dtype  = np.float32))
    # # print('predict_on_batch out: ', predict_on_batch_out)
    # #%% 10
    # # show likeliood
    # data = pd.DataFrame(np.zeros((2000, 7), 
    #                     dtype = np.float32), 
    #                     columns = ['v', 'a', 'z', 't', 'theta', 'rt', 'choice'])
    # data['v'] = 0.5 # drift rate
    # data['a'] = 0.75 # threshold
    # data['z'] = 0.5 # bias
    # data['t'] = 0.2 # nondesicion time
    # # data['theta'] = 0.1 # linear boundary, the angle of boundary
    # data['rt'].iloc[:1000] = np.linspace(5, 0, 1000)
    # data['rt'].iloc[1000:] = np.linspace(0, 5, 1000)
    # data['choice'].iloc[:1000] = -1
    # data['choice'].iloc[1000:] = 1
    # print(data['choice'])
    # #%% 11
    # # Network predictions
    # predict_on_batch_out = network.predict_on_batch(data.values.astype(np.float32))

    # # Simulations
    # from ssms.basic_simulators import simulator 
    # sim_out = simulator(model = 'angle', 
    #                     theta = data.values[0, :-2],
    #                     n_samples = 2000)
    # #%% 12
    # data_rt = np.array(data['rt'] * data['choice']);
    # # Plot network predictions
    # # plt.plot(data['rt'] * data['choice'], np.exp(predict_on_batch_out), color = 'black', label = 'network')
    # plt.plot(data_rt, np.exp(predict_on_batch_out), color = 'black', label = 'network')
    # # Plot simulations
    # plt.hist(sim_out['rts'] * sim_out['choices'], bins = 30, histtype = 'step', label = 'simulations', color = 'blue', density  = True)
    # plt.legend()
    # plt.title('SSM likelihood')
    # plt.xlabel('rt')
    # plt.ylabel('likelihod')
