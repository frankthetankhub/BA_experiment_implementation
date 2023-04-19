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
        metrics = np.zeros((self.num_epochs, 4))

        #save model weight init
        np.savez_compressed(self.save_filename + "_initial_weights.npz", self.model.get_weights()['w'])
        np.savez_compressed(self.save_filename + "_initial_biases.npz", self.model.get_weights()['b'])
        for epoch in range(1, self.num_epochs + 1):
            logging.info("beginning epoch {:d}".format(self.epoch + epoch))
            if self.monitor:
                self.monitor.start_monitor()
            start_train = datetime.datetime.now()
            for j in range(self.data.x_train.shape[0] // self.data.batch_size):#wichtige stelle hier startet der trainingsprocess für single_process bzw parallel training
                start_pos = j * self.data.batch_size
                end_pos = (j + 1) * self.data.batch_size
                batch = self.data.x_train[start_pos:end_pos], self.data.y_train[start_pos:end_pos]
                self.update = self.model.train_on_batch(x=batch[0], y=batch[1])

                self.model.apply_update(self.update)
            end_train = datetime.datetime.now()
            trainig_time = (end_train - start_train).total_seconds()
            if self.monitor:
                self.monitor.stop_monitor()

            if testing:
                t3 = datetime.datetime.now()
                accuracy_test, activations_test = self.model.predict(self.data.x_test, self.data.y_test)
                accuracy_train, activations_train = self.model.predict(self.data.x_train, self.data.y_train)
                t4 = datetime.datetime.now()
                self.maximum_accuracy = max(self.maximum_accuracy, accuracy_test)
                loss_test = self.model.compute_loss(self.data.y_test, activations_test)
                loss_train = self.model.compute_loss(self.data.y_train, activations_train)
                metrics[epoch-1, 0] = loss_train
                metrics[epoch-1, 1] = loss_test
                metrics[epoch-1, 2] = accuracy_train
                metrics[epoch-1, 3] = accuracy_test
                testing_time = (t4 - t3)
                self.logger.info(f"Testing time: {testing_time}\n; Training time: {trainig_time}\n; Loss train: {loss_train}; Loss test: {loss_test}; \n"
                                 f"Accuracy train: {accuracy_train}; Accuracy test: {accuracy_test}; \n"
                                 f"Maximum accuracy test: {self.maximum_accuracy}")
                self.validate_time += testing_time.total_seconds()
                # save performance metrics values in a file
                if self.save_filename != "":
                    np.savetxt(self.save_filename + ".txt", metrics)

            if self.stop_training:
                break

            if epoch % self.save_weight_interval == 0:
                weights.append(self.model.get_weights()['w'].copy())
                biases.append(self.model.get_weights()['b'].copy()) #wichtige stelle da hier potentiell das saven von weights etc angebracht ist
            if epoch < self.num_epochs - 1:  # do not change connectivity pattern after the last epoch
                self.model.weight_evolution(epoch) #wichtige stelle da hier weight evolution durchgeführt wird
                self.weights = self.model.get_weights()

        logging.info("Signing off")
        np.savez_compressed(self.save_filename + "_weights.npz", *weights)
        np.savez_compressed(self.save_filename + "_biases.npz", *biases)

        if self.save_filename != "" and self.monitor:
            with open(self.save_filename + "_monitor.json", 'w') as file:
                file.write(json.dumps(self.monitor.get_stats(), indent=4, sort_keys=True, default=str))

    def validate(self):
        t3 = datetime.datetime.now()
        accuracy_test, activations_test = self.model.predict(self.data.x_test, self.data.y_test)
        accuracy_train, activations_train = self.model.predict(self.data.x_train, self.data.y_train)
        t4 = datetime.datetime.now()
        self.maximum_accuracy = max(self.maximum_accuracy, accuracy_test)
        loss_test = self.model.compute_loss(self.data.y_test, activations_test)
        loss_train = self.model.compute_loss(self.data.y_train, activations_train)
        self.logger.info(f"Testing time: {t4 - t3}\n; Loss train: {loss_train}; Loss test: {loss_test}; \n"
                         f"Accuracy train: {accuracy_train}; Accuracy test: {accuracy_test}; \n"
                         f"Maximum accuracy test: {self.maximum_accuracy}")