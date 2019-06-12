#Main file
import srez_input
import srez_model
import srez_train

import os.path
import random
import numpy as np
import numpy.random

import tensorflow as tf
import shutil, os, errno # utils handling file manipulation

from scipy import io as sio #.mat I/O

FLAGS = tf.app.flags.FLAGS

real_img = 0

def mkdirp(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def prepare_dirs(delete_train_dir=False, shuffle_filename=True):
    # Create checkpoint dir (do not delete anything)
    if not tf.gfile.Exists(FLAGS.checkpoint_dir):
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)
    
    # Cleanup train dir
    if delete_train_dir:
        try:
            if tf.gfile.Exists(FLAGS.train_dir):
                tf.gfile.DeleteRecursively(FLAGS.train_dir)
            tf.gfile.MakeDirs(FLAGS.train_dir)
        except:
            try:
                shutil.rmtree(FLAGS.train_dir)
            except:
                print('fail to delete train dir {0} using tf.gfile, will use shutil'.format(FLAGS.train_dir))
            mkdirp(FLAGS.train_dir)


    # Return names of training files
    
    if not tf.gfile.Exists(FLAGS.dataset_train) or \
       not tf.gfile.IsDirectory(FLAGS.dataset_train):
        raise FileNotFoundError("Could not find folder `%s'" % (FLAGS.dataset_train,))
    

    filenames = tf.gfile.ListDirectory(FLAGS.dataset_train)
    filenames = sorted(filenames)
    if shuffle_filename:
        random.shuffle(filenames)
    filenames = [os.path.join(FLAGS.dataset_train, f) for f in filenames]

    return filenames

def get_filenames(dir_file='', shuffle_filename=False):
    try:
        filenames = tf.gfile.ListDirectory(dir_file)
    except:
        print('cannot get files from {0}'.format(dir_file))
        return []
    filenames = sorted(filenames)
    if shuffle_filename:
        random.shuffle(filenames)
    else:
        filenames = sorted(filenames)
    filenames = [os.path.join(dir_file, f) for f in filenames if f.endswith('.jpg')]
    return filenames



def setup_tensorflow(gpu_memory_fraction=1):
    # Create session
    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    if FLAGS.gpu_memory_fraction>0:
        config.gpu_options.per_process_gpu_memory_fraction = min(gpu_memory_fraction, FLAGS.gpu_memory_fraction)
    else:
        config.gpu_options.per_process_gpu_memory_fraction = min(1.0, -FLAGS.gpu_memory_fraction)
    sess = tf.Session(config=config)
    print('TF session setup for gpu usage cap of {0}'.format(config.gpu_options.per_process_gpu_memory_fraction))

    # Initialize rng with a deterministic seed
    with sess.graph.as_default():
        tf.set_random_seed(FLAGS.random_seed)
    
    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)

# SummaryWriter is deprecated
# tf.summary.FileWriter.
    #summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

    return sess, None  #summary_writer   

def _demo():
    # Load checkpoint
    if not tf.gfile.IsDirectory(FLAGS.checkpoint_dir):
        raise FileNotFoundError("Could not find folder `%s'" % (FLAGS.checkpoint_dir,))

    # Setup global tensorflow state
    sess, summary_writer = setup_tensorflow()

    # Prepare directories
    filenames = prepare_dirs(delete_train_dir=False)

    # Setup async input queues
    features, labels, masks = srez_input.setup_inputs(sess, filenames)

    # Create and initialize model
    [gene_minput, gene_moutput,
     gene_output, gene_var_list,
     disc_real_output, disc_fake_output, disc_var_list,train_phase,z_val] = \
            srez_model.create_model(sess, features, labels, masks)

    # Restore variables from checkpoint
    saver = tf.train.Saver()
    filename = 'checkpoint_new.txt'
    filename = os.path.join(FLAGS.checkpoint_dir, filename)
    saver.restore(sess, filename)

    # Execute demo
    srez_demo.demo1(sess)

class TrainData(object):
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)

def _train():
    # Setup global tensorflow state
    sess, _oldwriter = setup_tensorflow()

    # image_size
    if FLAGS.sample_size_y>0:
        image_size = [FLAGS.sample_size, FLAGS.sample_size_y]
    else:
        image_size = [FLAGS.sample_size, FLAGS.sample_size]

    # Prepare train and test directories (SEPARATE FOLDER)
    prepare_dirs(delete_train_dir=True, shuffle_filename=False)
    filenames_input_train = get_filenames(dir_file=FLAGS.dataset_train, shuffle_filename=True)
    filenames_output_train = get_filenames(dir_file=FLAGS.dataset_train, shuffle_filename=True)
    filenames_input_test = get_filenames(dir_file=FLAGS.dataset_test, shuffle_filename=False)
    filenames_output_test = get_filenames(dir_file=FLAGS.dataset_test, shuffle_filename=False)




    # check input and output sample number matches (SEPARATE FOLDER)
    assert(len(filenames_input_train)==len(filenames_output_train))
    num_filename_train = len(filenames_input_train)
    assert(len(filenames_input_test)==len(filenames_output_test))
    num_filename_test = len(filenames_input_test)


    if FLAGS.permutation_split:
        index_permutation_split = random.sample(num_filename_train, num_filename_train)
        filenames_input_train = [filenames_input_train[x] for x in index_permutation_split]
        filenames_output_train = [filenames_output_train[x] for x in index_permutation_split]
        #print(np.shape(filenames_input_train))

    if FLAGS.permutation_split:
        index_permutation_split = random.sample(num_filename_test, num_filename_test)
        filenames_input_test = [filenames_input_test[x] for x in index_permutation_split]
        filenames_output_test = [filenames_output_test[x] for x in index_permutation_split]

    # Separate training and test sets (SEPARATE FOLDERS)
    train_filenames_input = filenames_input_train[:FLAGS.sample_train]    
    train_filenames_output = filenames_output_train[:FLAGS.sample_train]  

    test_filenames_input  = filenames_input_test[:FLAGS.sample_test]
    test_filenames_output  = filenames_output_test[:FLAGS.sample_test]



    # randomly subsample for train
    if FLAGS.subsample_train > 0:

        index_sample_train_selected = random.sample(range(len(train_filenames_input)), FLAGS.subsample_train)
        if not FLAGS.permutation_train:
            index_sample_train_selected = sorted(index_sample_train_selected)
        train_filenames_input = [train_filenames_input[x] for x in index_sample_train_selected]
        train_filenames_output = [train_filenames_output[x] for x in index_sample_train_selected]
        print('randomly sampled {0} from {1} train samples'.format(len(train_filenames_input), len(filenames_input_train[:-FLAGS.sample_test])))

    # randomly sub-sample for test    
    if FLAGS.subsample_test > 0:
        index_sample_test_selected = random.sample(range(len(test_filenames_input)), FLAGS.subsample_test)
        print(len(test_filenames_input))
        print(FLAGS.subsample_test)
        if not FLAGS.permutation_test:
            index_sample_test_selected = sorted(index_sample_test_selected)
        test_filenames_input = [test_filenames_input[x] for x in index_sample_test_selected]
        test_filenames_output = [test_filenames_output[x] for x in index_sample_test_selected]
        print('randomly sampled {0} from {1} test samples'.format(len(test_filenames_input), len(test_filenames_input[:-FLAGS.sample_test])))

    #print('test_filenames_input',test_filenames_input)            

    # get undersample mask
    from scipy import io as sio
    try:
        content_mask = sio.loadmat(FLAGS.sampling_pattern)
        key_mask = [x for x in content_mask.keys() if not x.startswith('_')]
        mask = content_mask[key_mask[0]]
    except:
        mask = None

    print(len(train_filenames_input))
    print(len(train_filenames_output))
    print(len(test_filenames_input))
    print(len(test_filenames_output))

    mask = None

    # Setup async input queues
    train_features, train_labels, train_masks = srez_input.setup_inputs_one_sources(sess, train_filenames_input, train_filenames_output, 
                                                                        image_size=image_size, 
                                                                        # undersampling
                                                                        axis_undersample=FLAGS.axis_undersample, 
                                                                        r_factor=FLAGS.R_factor,
                                                                        r_alpha=FLAGS.R_alpha,
                                                                        r_seed=FLAGS.R_seed,
                                                                        sampling_mask=mask
                                                                        )
    test_features,  test_labels, test_masks  = srez_input.setup_inputs_one_sources(sess, test_filenames_input, test_filenames_output,
                                                                        image_size=image_size, 
                                                                        # undersampling
                                                                        axis_undersample=FLAGS.axis_undersample, 
                                                                        r_factor=FLAGS.R_factor,
                                                                        r_alpha=FLAGS.R_alpha,
                                                                        r_seed=FLAGS.R_seed,
                                                                        sampling_mask=mask
                                                                       
                                                                   )
    

    print('features_size', train_features.get_shape())
    print('labels_size', train_labels.get_shape())
    print('masks_size', train_masks.get_shape())


    # sample train and test
    num_sample_train = len(train_filenames_input)
    num_sample_test = len(test_filenames_input)
    print('train on {0} samples and test on {1} samples'.format(num_sample_train, num_sample_test))

    # Add some noise during training (think denoising autoencoders)
    noise_level = .00
    noisy_train_features = train_features + \
                           tf.random_normal(train_features.get_shape(), stddev=noise_level)

    # Create and initialize model
    [sing_vals,mn, sd, gene_minput, label_minput, gene_moutput, gene_moutput_list, \
     gene_output, gene_output_list, gene_var_list, gene_layers_list, gene_mlayers_list, gene_mask_list, gene_mask_list_0, \
     disc_real_output, disc_fake_output, disc_var_list, train_phase,print_bool,z_val,disc_layers, eta, nmse, kappa] = \
            srez_model.create_model(sess, noisy_train_features, train_labels, train_masks, architecture=FLAGS.architecture)

    
    gene_loss, gene_dc_loss, gene_ls_loss, gene_mse_loss, list_gene_losses, gene_mse_factor = srez_model.create_generator_loss(disc_fake_output, gene_output, gene_output_list, train_features, train_labels, train_masks, mn, sd)
    #disc_loss,disc_real_loss, disc_fake_loss,gradient_penalty = \
                     #srez_model.create_discriminator_loss(disc_real_output, disc_fake_output,real_data = tf.identity(train_labels), fake_data = tf.abs(gene_output))
    disc_real_loss, disc_fake_loss = \
                     srez_model.create_discriminator_loss(disc_real_output, disc_fake_output)
    disc_loss = tf.add(disc_real_loss, disc_fake_loss, name='disc_loss')
    
    (global_step, learning_rate, gene_minimize, disc_minimize) = \
            srez_model.create_optimizers(gene_loss, gene_var_list,
                                         disc_loss, disc_var_list)


    # tensorboard
    summary_op=tf.summary.merge_all()


    #restore variables from checkpoint
    filename = 'checkpoint_new.txt'
    filename = os.path.join(FLAGS.checkpoint_dir, filename)
    metafile=filename+'.meta'

    
    if tf.gfile.Exists(metafile):
        saver = tf.train.Saver()
        print("Loading checkpoint from file `%s'" % (filename,))
        saver.restore(sess, filename)
    else:
        print("No checkpoint `%s', train from scratch" % (filename,))
        sess.run(tf.global_variables_initializer())


    # Train model
    train_data = TrainData(locals())
    srez_train.train_model(sess,train_data, num_sample_train, num_sample_test)

def main(argv=None):
    # Training or showing off?

    if FLAGS.run == 'demo':
        _demo()
    elif FLAGS.run == 'train':
        _train()

if __name__ == '__main__':
  tf.app.run()

