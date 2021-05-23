import os
import numpy as np
import errno 
from IPython import display
from matplotlib import pyplot as plt
import torch
from tensorboardX import SummaryWriter

'''
    TensorBoard Data will be stored in './runs' path
'''


class Logger:

    def __init__(self, model_name, data_name):
        self.model_name = model_name
        self.data_name = data_name

        self.comment = '{}_{}'.format(model_name, data_name)
        self.data_subdir = '{}/{}'.format(model_name, data_name)

        # TensorBoard
        self.writer = SummaryWriter(comment=self.comment)
    def newLog(self,actor_loss,critic_loss, epoch):
        if isinstance(actor_loss, torch.autograd.Variable):
            actor_loss = actor_loss.data.cpu().numpy()
        if isinstance(critic_loss, torch.autograd.Variable):
            critic_loss = critic_loss.data.cpu().numpy()
        
        step = epoch

        self.writer.add_scalar(
            '{}/actor_error'.format(self.comment), actor_loss, step)
        self.writer.add_scalar(
            '{}/critic_error'.format(self.comment), critic_loss, step)




    def log(self, ae_error, class_error, adv_error, epoch, n_batch, num_batches, description='train'):

        # var_class = torch.autograd.variable.Variable
        if isinstance(ae_error, torch.autograd.Variable):
            ae_error = ae_error.data.cpu().numpy()
        if isinstance(class_error, torch.autograd.Variable):
            class_error = class_error.data.cpu().numpy()
        if isinstance(adv_error, torch.autograd.Variable):
            adv_error = adv_error.data.cpu().numpy()

        step = Logger._step(epoch, n_batch, num_batches)
        self.writer.add_scalar(
            '{}/ae_error_{}'.format(self.comment, description), ae_error, step)
        self.writer.add_scalar(
            '{}/class_error_{}'.format(self.comment, description), class_error, step)
        self.writer.add_scalar(
            '{}/adv_error_{}'.format(self.comment, description), adv_error, step)
            
    
    def save_model(self, model, epoch):
        out_dir = './data/models/{}'.format(self.data_subdir)
        Logger._make_dir(out_dir) 
        torch.save(model.state_dict(),
                   '{}/{}'.format(out_dir, self.data_name))


    def close(self):
        self.writer.close()

    @staticmethod
    def _step(epoch, n_batch, num_batches):
        return epoch * num_batches + n_batch

    @staticmethod
    def _make_dir(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
