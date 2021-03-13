
""" 
Model definitions.
"""

import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import networks
import common
import data_utils
import train_utils
import wandb
import metrics


class Trainer:

    def __init__(self, args):
        self.args = args
        self.config, self.output_dir, self.logger, self.device = common.init_experiment(args, seed=420)
        self.train_loader, self.train_data, self.val_data = data_utils.get_dataloaders(
            args['root'], self.config['dataloader'], self.config['model']['horizon'], self.device
        )
        self.train_data = self.train_data.view(-1,)
        self.val_data = self.val_data.view(-1,)

        # Models, optimizer and scheduler
        self.model = networks.Forecaster(self.config['model']).to(self.device)
        self.optim = train_utils.get_optimizer(
            config = self.config['optim'], 
            params = self.model.parameters()
        )
        self.scheduler, self.warmup_epochs = train_utils.get_scheduler(
            config = {**self.config['scheduler'], 'epochs': self.config['epochs']}, 
            optimizer = self.optim
        )

        # Count model parameters
        total_params = common.count_parameters([self.model])
        if total_params // 1e06 > 0:
            self.logger.record(f'Total trainable parameters: {round(total_params/1e06, 2)}M', mode='info')
        else:
            self.logger.record(f'Total trainable parameters: {total_params}', mode='info')
        
        # Criterion and logging
        self.criterion = nn.MSELoss()
        self.best_val = 1e09
        self.best_mape = 10.0
        self.done_epochs = 0
        run = wandb.init('stock-prediction-nbeats')
        self.logger.write(run.get_url(), mode='info')

        # Warmup handling
        if self.warmup_epochs > 0:
            self.warmup_rate = self.optim.param_groups[0]['lr']/self.warmup_epochs

        # Load best model if specified
        if args['load'] is not None:
            if os.path.exists(os.path.join(args['load'], 'best_model.ckpt')):
                self.load_model(args['load'])
                self.logger.record(f"Successfully loaded saved model from {args['load']}", mode='info')
            else:
                raise NotImplementedError(f"Could not load best_model.ckpt from {args['load']}; please check your path", mode='info')

        self.horizon = self.config['model']['horizon']
        self.lookback = self.config['model']['lookback_horizon_ratio'] * self.horizon

    def train_on_batch(self, batch):
        inp, trg = batch
        out = self.model(inp)
        loss = self.criterion(out, trg)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return {'Loss': loss.item(), 'MAPE': metrics.MAPE(out, trg)}

    def validate(self): 
        self.model.eval()
        history = self.train_data.unsqueeze(0)
        output = []
        for i in range(self.config['model']['forecast_horizon_ratio']):
            inp = history[:, -self.lookback:]
            with torch.no_grad():
                out = self.model(inp)
            output.append(out)
            history = torch.cat((history, out), dim=-1)
        out = torch.cat(output, dim=-1).squeeze(0)
        loss = self.criterion(out, self.val_data)
        return {'Loss': loss.item(), 'MAPE': metrics.MAPE(out, self.val_data)}, out

    def save_state(self, epoch):
        state = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optim': self.optim.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None
        }
        torch.save(state, os.path.join(self.output_dir, 'last_state.ckpt'))

    def load_state(self):
        state_path = os.path.join(self.output_dir, 'last_state.ckpt')
        state = torch.load(state_path)
        self.model.load_state_dict(state['model'])
        self.optim.load_state_dict(state['optim'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state['scheduler'])
        return state['epoch']

    def save_model(self):
        state = {'model': self.model.state_dict()}
        torch.save(state, os.path.join(self.output_dir, 'best_model.ckpt'))

    def load_model(self, path):
        state = torch.load(path)
        self.model.load_state_dict(state['model'])
        
    def adjust_learning_rate(self, epoch):
        if epoch < self.warmup_epochs:
            for group in self.optim.param_groups:
                group['lr'] = 1e-12 + (epoch * self.warmup_rate)
        else:
            self.scheduler.step()

    def compare_predictions(self, forecast):
        fig = plt.figure(figsize=(10, 6))
        plt.plot(self.val_data.detach().cpu().numpy(), color='blue', alpha=0.8, label='Actual')
        plt.plot(forecast.detach().cpu().numpy(), color='red', alpha=0.8, label='Forecast')
        plt.grid(alpha=0.4)
        plt.legend()
        plt.xlabel('Data index')
        if self.args['task'] == 'temp':
            plt.ylabel('Temperature (Celsius)')
        else:
            plt.ylabel('Precipitation (mm)')
        wandb.log({"Forecast comparison": [wandb.Image(fig, caption="Actual vs forecasted")]})

    def stack_breakdown(self, run_id):
        self.load_model(os.path.join(self.output_dir, 'best_model.ckpt'))
        self.model.eval()
        history = self.train_data.unsqueeze(0)
        stackwise_frc = {}
        output = []
        for i in range(self.config['model']['forecast_horizon_ratio']):
            inp = history[:, -self.lookback:]
            with torch.no_grad():
                out, stack_frc = self.model(inp, return_stack_frc=True)
            output.append(out)
            history = torch.cat((history, out), dim=-1)[:, -self.lookback:]
            for k, v in stack_frc.items():
                if k in stackwise_frc.keys():
                    stackwise_frc[k].extend(stack_frc[k])
                else:
                    stackwise_frc[k] = stack_frc[k]
        
        out = torch.cat(output, dim=-1).squeeze(0)
        plt.figure(figsize=(15, 6))
        plt.plot(self.val_data.detach().cpu().numpy(), color='blue', alpha=0.8, label='Actual')
        plt.plot(out.detach().cpu().numpy().reshape(-1,), color='red', alpha=0.8, label='Forecasted')
        plt.grid(alpha=0.4)
        plt.legend()
        plt.xlabel('Data index')
        if self.args['task'] == 'temp':
            plt.ylabel('Temperature (Celsius)')
        else:
            plt.ylabel('Precipitation (mm)')
        plt.savefig(os.path.join(self.output_dir, f'test_prediction_{run_id}.png'), pad_inches=0.05)

        # Plot the sequences against val data
        for k in stackwise_frc.keys():
            plt.figure(figsize=(15, 6))
            plt.plot(self.val_data.detach().cpu().numpy(), color='blue', alpha=0.5, label='Actual')
            plt.plot(stackwise_frc[k], color='red', alpha=0.8, label=k)
            plt.grid(alpha=0.4)
            plt.legend()
            plt.xlabel('Data index')
            if self.args['task'] == 'temp':
                plt.ylabel('Temperature (Celsius)')
            else:
                plt.ylabel('Precipitation (mm)')
            plt.savefig(os.path.join(self.output_dir, f'{k}_{run_id}.png'), pad_inches=0.05)

    def autoregressive_forecast(self, run_id):
        self.load_model(os.path.join(self.output_dir, 'best_model.ckpt'))
        self.model.eval()
        total_points = len(self.train_data) + len(self.val_data)
        history = self.train_data.unsqueeze(0)[:, :600]
        output = []
        while len(output) < (total_points - 600):
            inp = history[:, -self.lookback:]
            with torch.no_grad():
                out = self.model(inp)
            output.extend(out.detach().cpu().numpy().reshape(-1,).tolist())
            history = torch.cat((history, out), dim=-1)

        all_data = self.train_data.detach().cpu().numpy().reshape(-1,).tolist() + self.val_data.detach().cpu().numpy().reshape(-1,).tolist()
        inp_data = self.train_data.unsqueeze(0)[:, :600].detach().cpu().numpy().reshape(-1,).tolist()

        plt.figure(figsize=(25, 6))
        plt.plot(np.arange(total_points), all_data, color='blue', alpha=0.7, label='Actual')
        plt.plot(np.arange(600, total_points), output, color='red', alpha=0.7, label='Forecast')
        plt.axvline(x=600, ymin=0, ymax=1, color='gray', linestyle='--', label='Forecast begins')
        plt.xlabel("Data index")
        if self.args['task'] == 'temp':
            plt.ylabel('Temperature (Celsius)')
        else:
            plt.ylabel('Precipitation (mm)')
        plt.legend()
        plt.grid(alpha=0.4)
        plt.savefig(os.path.join(self.output_dir, f'autoregressive_forecast_{run_id}.png'), pad_inches=0.05)

    def train(self, run_id=0):
        print()
        for epoch in range(self.done_epochs, self.config['epochs']+1):
            train_meter = common.AverageMeter()
            val_meter = common.AverageMeter()
            self.train_data, self.val_data = self.train_data.to(self.device), self.val_data.to(self.device)
            self.logger.record('Epoch [{:3d}/{}]'.format(epoch, self.config['epochs']), mode='train')
            if self.scheduler is not None:
                self.adjust_learning_rate(epoch+1)

            for idx in range(len(self.train_loader)):
                batch = self.train_loader.flow()
                train_metrics = self.train_on_batch(batch)
                wandb.log({'Loss': train_metrics['Loss'], 'MAPE': train_metrics['MAPE'], 'Epoch': epoch})
                train_meter.add(train_metrics)
                common.progress_bar(progress=idx/len(self.train_loader), status=train_meter.return_msg())

            common.progress_bar(progress=1, status=train_meter.return_msg())
            self.logger.write(train_meter.return_msg(), mode='train')
            wandb.log({'Learning rate': self.optim.param_groups[0]['lr'], 'Epoch': epoch})

            # Save state
            self.save_state(epoch)

            # Validation
            if epoch % self.config['eval_every'] == 0:
                self.logger.record('Epoch [{:3d}/{}]'.format(epoch, self.config['epochs']), mode='val')
                val_metrics, forecast = self.validate()
                self.compare_predictions(forecast)
                val_meter.add(val_metrics)

                self.logger.record(val_meter.return_msg(), mode='val')
                val_metrics = val_meter.return_metrics()
                wandb.log({'Validation loss': val_metrics['Loss'], 'Validation MAPE': val_metrics['MAPE'], 'Epoch': epoch})

                if val_metrics['Loss'] < self.best_val:
                    self.best_val = val_metrics['Loss']
                    self.save_model()

                if val_metrics['MAPE'] < self.best_mape:
                    self.best_mape = val_metrics['MAPE']

        self.stack_breakdown(run_id)
        self.autoregressive_forecast(run_id)
        print('\n\n[INFO] Training complete!')
        return self.best_val, self.best_mape