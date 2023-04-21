import numpy as np
import logging
from wasap_sgd.mpi.process import MPIWorker, MPIMaster
import datetime
import json


class MPISingleWorker(MPIWorker):
    """This class trains its model with no communication to other processes"""
    def __init__(self, num_epochs, data, algo, model,monitor, save_filename, save_weight_interval=20):

        self.has_parent = False
        self.best_val_loss = None
        self.save_weight_interval = save_weight_interval

        super(MPISingleWorker, self).__init__(data, algo, model, process_comm=None, parent_comm=None,
                                              parent_rank=None, num_epochs=num_epochs, monitor=monitor,
                                              save_filename=save_filename)

    def train(self, testing=True):
        self.check_sanity()

        weights = []
        biases = []

        self.maximum_accuracy = 0
        #include training per epoch time as metric
        metrics = np.zeros((self.num_epochs, 5))

        #save model weight init
        np.savez_compressed(self.save_filename + "_initial_weights.npz", self.model.get_weights()['w'])
        np.savez_compressed(self.save_filename + "_initial_biases.npz", self.model.get_weights()['b'])
        batches_per_epoch = self.data.get_train_data().shape[0] // self.data.batch_size
        best_model = None
        for epoch in range(1, self.num_epochs + 1):
            logging.info("beginning epoch {:d}".format(self.epoch + epoch))
            if self.monitor:
                self.monitor.start_monitor()

            num_batches = 0
            start_train = datetime.datetime.now()
            for x_b, y_b in self.data.generate_data():#wichtige stelle hier startet der trainingsprocess für single_process bzw parallel training
                num_batches += 1
                self.update = self.model.train_on_batch(x=x_b, y=y_b)

                self.model.apply_update(self.update)
                if num_batches % batches_per_epoch == 0:
                    break
            
            end_train = datetime.datetime.now()
            training_time = (end_train - start_train).total_seconds()

            if self.monitor:
                self.monitor.stop_monitor()

            if testing:
                t3 = datetime.datetime.now()
                accuracy_test, activations_test = self.model.predict(self.data.get_test_data(), self.data.get_test_labels())
                accuracy_train, activations_train = self.model.predict(self.data.get_train_data(), self.data.get_train_labels())
                # accuracy_test, activations_test = self.model.predict(self.data.x_test, self.data.y_test)
                # accuracy_train, activations_train = self.model.predict(self.data.x_train, self.data.y_train)
                t4 = datetime.datetime.now()
                if accuracy_test > self.maximum_accuracy:
                    best_model_weights = self.model.get_weights()['w'].copy()
                    best_model_biases = self.model.get_weights()['b'].copy()
                    self.maximum_accuracy = accuracy_test
                #self.maximum_accuracy = max(self.maximum_accuracy, accuracy_test)
                # loss_test = self.model.compute_loss(self.data.y_test, activations_test)
                # loss_train = self.model.compute_loss(self.data.y_train, activations_train)
                loss_test = self.model.compute_loss(self.data.get_test_labels(), activations_test)
                loss_train = self.model.compute_loss(self.data.get_train_labels(), activations_train)
                metrics[epoch-1, 0] = loss_train
                metrics[epoch-1, 1] = loss_test
                metrics[epoch-1, 2] = accuracy_train
                metrics[epoch-1, 3] = accuracy_test
                metrics[epoch-1, 4] = training_time
                testing_time = (t4 - t3).total_seconds()
                self.logger.info(f"Training time: {training_time};")
                self.logger.info(f"Testing time: {testing_time}; \nLoss train: {loss_train}; Loss test: {loss_test}; \n"
                                 f"Accuracy train: {accuracy_train}; Accuracy test: {accuracy_test}; \n"
                                 f"Maximum accuracy test: {self.maximum_accuracy}")
                self.validate_time += testing_time
                # save performance metrics values in a file
                if self.save_filename != "":
                    np.savetxt(self.save_filename + ".txt", metrics)

            if self.stop_training:
                break

            if epoch % self.save_weight_interval == 0:
                weights.append(self.model.get_weights()['w'].copy())
                biases.append(self.model.get_weights()['b'].copy()) #wichtige stelle da hier potentiell das saven von weights etc angebracht ist
            if epoch < self.num_epochs - 1:  # do not change connectivity pattern after the last epoch
                start_evolution = datetime.datetime.now()
                self.model.weight_evolution(epoch) #wichtige stelle da hier weight evolution durchgeführt wird
                self.weights = self.model.get_weights()
                evolution_time = (datetime.datetime.now()- start_evolution).total_seconds()
                self.logger.info(f"Weights evolution time  {evolution_time}")
                self.evolution_time += evolution_time

        logging.info("Signing off")
        np.savez_compressed(self.save_filename + "_weights.npz", *weights)
        np.savez_compressed(self.save_filename + "_biases.npz", *biases)
        np.savez_compressed(self.save_filename + "_bestmodel_weights.npz", best_model_weights)
        np.savez_compressed(self.save_filename + "_bestmodel_biases.npz", best_model_biases)

        if self.save_filename != "" and self.monitor:
            with open(self.save_filename + "_monitor.json", 'w') as file:
                file.write(json.dumps(self.monitor.get_stats(), indent=4, sort_keys=True, default=str))

    def validate(self):
        t3 = datetime.datetime.now()
        accuracy_test, activations_test = self.model.predict(self.data.get_test_data(), self.data.get_test_labels())
        accuracy_train, activations_train = self.model.predict(self.data.get_train_data(), self.data.get_train_labels())
        t4 = datetime.datetime.now()
        self.maximum_accuracy = max(self.maximum_accuracy, accuracy_test)
        loss_test = self.model.compute_loss(self.data.get_test_labels(), activations_test)
        loss_train = self.model.compute_loss(self.data.get_train_labels(), activations_train)
        t_5 = (t4-t3).total_seconds()
        self.logger.info(f"Testing time: {t_5}; \nLoss train: {loss_train}; Loss test: {loss_test}; \n"
                         f"Accuracy train: {accuracy_train}; Accuracy test: {accuracy_test}; \n"
                         f"Maximum accuracy test: {self.maximum_accuracy}")