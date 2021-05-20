import tensorflow as tf
import os
import argparse
from tacotron.feeder import Feeder
from hparams import hparams as hparamspy
from tacotron.models import create_model
from tqdm import tqdm
import numpy as np

_batches_per_group = 64


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='')
    parser.add_argument('--hparams', default='', help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--model', default='Tacotron-2')
    parser.add_argument('--tacotron_input', default='training_data/train.txt')
    parser.add_argument('--nb_speaker', default=96, help='Number of speaker during training.')
    parser.add_argument('--embedding_dir', default="logs-speaker_embeddings", help='Directory to save the speaker embeddings.')
    parser.add_argument('--debug', default=False, help='Print debugging information')
    parser.add_argument('--verbose', default=True, help='Print progress information')
    args = parser.parse_args()

    debug_flag = args.debug
    verbose_flag = args.verbose

    run_name = args.model
    log_dir = os.path.join(args.base_dir, 'logs-{}'.format(run_name))
    input_path = os.path.join(args.base_dir, args.tacotron_input)
    hparams = hparamspy.parse(args.hparams)
    tensorboard_dir = os.path.join(log_dir, 'tacotron_events')
    save_dir = os.path.join(log_dir, 'taco_pretrained')

    # Set up data feeder
    coord = tf.train.Coordinator()
    with tf.variable_scope('datafeeder') as scope:
        feeder = Feeder(coord, input_path, hparams, split=False)

    # Create model
    global_step = tf.Variable(0, name='global_step', trainable=False)
    with tf.variable_scope('Tacotron_model', reuse=tf.AUTO_REUSE) as scope:
        model = create_model("Tacotron", hparams)
        initialize_args = {"inputs": feeder.inputs, "input_lengths": feeder.input_lengths,
                           "mel_targets": feeder.mel_targets, "stop_token_targets": feeder.token_targets,
                           "targets_lengths": feeder.targets_lengths, "global_step": global_step, "is_training": False,
                           "split_infos": feeder.split_infos}
        if hparams.predict_linear:
            initialize_args["linear_targets"] = feeder.linear_targets
        if hparams.tacotron_reference_waveform:
            initialize_args["mel_references"] = feeder.mel_references
            initialize_args["nb_sample"] = len(feeder._metadata)
        if hparams.tacotron_multi_speaker:
            initialize_args["speaker_id_target"] = feeder.speaker_id_target
            initialize_args["nb_speaker"] = args.nb_speaker
        model.initialize(**initialize_args)

    saver = tf.train.Saver(max_to_keep=5)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:
        summary_writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)

        sess.run(tf.global_variables_initializer())

        # Restore saved model
        checkpoint_state = tf.train.get_checkpoint_state(save_dir)
        saver.restore(sess, checkpoint_state.model_checkpoint_path)

        # Embeddings speaker metadata
        os.makedirs(args.embedding_dir, exist_ok=False)
        speaker_embedding_meta = os.path.join(args.embedding_dir, 'SpeakerEmbeddings.tsv')
        with open(speaker_embedding_meta, 'w', encoding='utf-8') as f:
            f.write("Filename\tSpeaker\n") # Header

            n = feeder._hparams.tacotron_batch_size
            r = feeder._hparams.outputs_per_step
            speaker_embeddings = []
            examples = []

            if debug_flag:
                print(len(feeder._train_meta))
                print(len(feeder._train_meta[0]))
                print(n * _batches_per_group)

            # Extract speaker label and embedding
            for i in range(n * _batches_per_group):
                # if i<10:
                if i<len(feeder._train_meta):
                    example = feeder._get_next_example()
                    metadata = feeder._train_meta[i]
                    f.write('{}\t{}\n'.format(metadata[1], metadata[-1]))
                    examples.append(example)

                    batch = [example]
                    feed_dict = dict(zip(feeder._placeholders, feeder._prepare_batch(batch, r)))
                    sess.run(feeder._enqueue_op, feed_dict=feed_dict)
                    speaker_embedding = sess.run([model.embedding_speaker])
                    speaker_embeddings.append(speaker_embedding)

                    if verbose_flag:
                        print("\r\r\r\r\r\r\r\r{}/{}".format(i, len(feeder._train_meta)), end=" ")
            if verbose_flag:
                print(" ")

        # Reshape the embeddings data
        speaker_embeddings = np.array(speaker_embeddings)
        if debug_flag:
            print(speaker_embeddings.shape)
        speaker_embeddings = speaker_embeddings.reshape((-1, 64))
        if debug_flag:
            print(speaker_embeddings.shape)


        # Save embeddings data for Tensorboard
        spk_emb = tf.Variable(speaker_embeddings, name='speaker_embeddings')
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver([spk_emb])

            sess.run(spk_emb.initializer)
            saver.save(sess, os.path.join(args.embedding_dir, 'speaker_embeddings.ckpt'))

            config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
            # One can add multiple embeddings.
            embedding = config.embeddings.add()
            embedding.tensor_name = spk_emb.name
            # Link this tensor to its metadata file (e.g. labels).
            embedding.metadata_path = 'SpeakerEmbeddings.tsv'
            # Saves a config file that TensorBoard will read during startup.
            tf.contrib.tensorboard.plugins.projector.visualize_embeddings(tf.summary.FileWriter("logs-speaker_embeddings"), config)


if __name__ == '__main__':
    main()
