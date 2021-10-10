"""Taken from Torch-XLA"""
import time
from datetime import datetime

class RateTracker(object):

  def __init__(self, smooth_factor=0.4):
    self._smooth_factor = smooth_factor
    self._start_time = time.time()
    self._partial_time = self._start_time
    self._partial_count = 0.0
    self._partial_rate = None
    self._count = 0.0

  def _update(self, now, rate):
    self._partial_count += self._count
    self._count = 0.0
    self._partial_time = now
    self._partial_rate = rate

  def add(self, count):
    self._count += count

  def _smooth(self, current_rate):
    if self._partial_rate is None:
      smoothed_rate = current_rate
    else:
      smoothed_rate = ((1 - self._smooth_factor) * current_rate +
                       self._smooth_factor * self._partial_rate)
    return smoothed_rate

  def rate(self):
    now = time.time()
    delta = now - self._partial_time
    report_rate = 0.0
    if delta > 0:
      report_rate = self._smooth(self._count / delta)
      self._update(now, report_rate)
    return report_rate

  def global_rate(self):
    delta = time.time() - self._start_time
    count = self._partial_count + self._count
    return count / delta if delta > 0 else 0.0


def now(format='%H:%M:%S'):
  return datetime.now().strftime(format)


def print_train_update(device,
                  step,
                  loss,
                  rate,
                  global_rate,
                  epoch=None,
                  summary_writer=None):
  """Prints the training metrics at a given step.
  Args:
    device (torch.device): The device where these statistics came from.
    step_num (int): Current step number.
    loss (float): Current loss.
    rate (float): The examples/sec rate for the current batch.
    global_rate (float): The average examples/sec rate since training began.
    epoch (int, optional): The epoch number.
    summary_writer (SummaryWriter, optional): If provided, this method will
      write some of the provided statistics to Tensorboard.
  """
  update_data = [
      'Training', 'Device={}'.format(device),
      'Epoch={}'.format(epoch) if epoch is not None else None,
      'Step={}'.format(step), 'Loss={:.5f}'.format(loss),
      'Rate={:.2f}'.format(rate), 'GlobalRate={:.2f}'.format(global_rate),
      'Time={}'.format(now())
  ]
  print('|', ' '.join(item for item in update_data if item), flush=True)

def print_test_update(device, accuracy, step=None):
  """Prints single-core test metrics.
  Args:
    device: Instance of `torch.device`.
    accuracy: Float.
  """
  update_data = [
      'Test', 'Device={}'.format(device),
      'Step={}'.format(step) if step is not None else None,
      'Accuracy={:.2f}'.format(accuracy) if accuracy is not None else None,
      'Time={}'.format(now())
  ]
  print('|', ' '.join(item for item in update_data if item), flush=True)
