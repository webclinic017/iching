#
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import gym.spaces
#
from apps.drl.c03.e02.app_config import AppConfig
from apps.drl.c03.e02.input_wrapper import InputWrapper
from apps.drl.c03.e02.discriminator import Discriminator
from apps.drl.c03.e02.generator import Generator

class C03E02(object):
    LEARNING_RATE = 0.0001
    REPORT_EVERY_ITER = 100
    SAVE_IMAGE_EVERY_ITER = 1000

    def __init__(self):
        self.name = 'apps.drl.c03.c03_e02.C03E02'
        self.logger = gym.logger
        self.logger.set_level(gym.logger.INFO)

    def startup(self, args={}):
        '''
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--cuda", default=False, action='store_true',
            help="Enable cuda computation")
        args = parser.parse_args()
        '''
        cudas = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
        for ci in range(len(cudas)):
            print(torch.cuda.get_device_name(ci))

        device = torch.device("cuda:0")
        envs = [
            InputWrapper(gym.make(name))
            for name in ('Breakout-v0', 'AirRaid-v0', 'Pong-v0')
        ]
        input_shape = envs[0].observation_space.shape

        net_discr = Discriminator(input_shape=input_shape).to(device)
        net_gener = Generator(output_shape=input_shape).to(device)

        objective = nn.BCELoss()
        gen_optimizer = optim.Adam(
            params=net_gener.parameters(), lr=C03E02.LEARNING_RATE,
            betas=(0.5, 0.999))
        dis_optimizer = optim.Adam(
            params=net_discr.parameters(), lr=C03E02.LEARNING_RATE,
            betas=(0.5, 0.999))

        gen_losses = []
        dis_losses = []
        iter_no = 0

        true_labels_v = torch.ones(AppConfig.BATCH_SIZE, device=device)
        fake_labels_v = torch.zeros(AppConfig.BATCH_SIZE, device=device)

        for batch_v in C03E02.iterate_batches(envs):
            # fake samples, input is 4D: batch, filters, x, y
            gen_input_v = torch.FloatTensor(
                AppConfig.BATCH_SIZE, Generator.LATENT_VECTOR_SIZE, 1, 1).to(device)
            gen_input_v.normal_(0, 1).to(device)
            batch_v = batch_v.to(device)
            gen_output_v = net_gener(gen_input_v)

            # train discriminator
            dis_optimizer.zero_grad()
            dis_output_true_v = net_discr(batch_v)
            dis_output_fake_v = net_discr(gen_output_v.detach())
            dis_loss = objective(dis_output_true_v, true_labels_v) + \
                    objective(dis_output_fake_v, fake_labels_v)
            dis_loss.backward()
            dis_optimizer.step()
            dis_losses.append(dis_loss.item())

            # train generator
            gen_optimizer.zero_grad()
            dis_output_v = net_discr(gen_output_v)
            gen_loss_v = objective(dis_output_v, true_labels_v)
            gen_loss_v.backward()
            gen_optimizer.step()
            gen_losses.append(gen_loss_v.item())

            iter_no += 1
            self.logger.info("Iter %d: gen_loss=%.3e, dis_loss=%.3e",
                        iter_no, gen_loss_v.item(),
                        dis_loss.item())
            if iter_no % C03E02.REPORT_EVERY_ITER == 0:
                self.logger.info("Iter %d: gen_loss=%.3e, dis_loss=%.3e",
                        iter_no, np.mean(gen_losses),
                        np.mean(dis_losses))
                gen_losses = []
                dis_losses = []
            if iter_no % C03E02.SAVE_IMAGE_EVERY_ITER == 0:
                self.logger.info('save...')

    @staticmethod
    def iterate_batches(envs, batch_size=AppConfig.BATCH_SIZE):
        batch = [e.reset() for e in envs]
        env_gen = iter(lambda: random.choice(envs), None)

        while True:
            e = next(env_gen)
            obs, reward, is_done, _ = e.step(e.action_space.sample())
            if np.mean(obs) > 0.01:
                batch.append(obs)
            if len(batch) == batch_size:
                # Normalising input between -1 to 1
                batch_np = np.array(batch, dtype=np.float32)
                batch_np *= 2.0 / 255.0 - 1.0
                yield torch.tensor(batch_np)
                batch.clear()
            if is_done:
                e.reset()