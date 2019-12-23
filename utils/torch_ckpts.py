#!/usr/bin/env python3
# Copyright 2018 Christian Henning
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
@title           :torch_ckpts.py
@author          :ch
@contact         :henningc@ethz.ch
@created         :11/18/2018
@version         :1.0
@python_version  :3.6.6

This module provides functions to handle PyTorch checkpoints with a similar
convenience as one might be used to in Tensorflow.
"""

import os
import torch
import time
import json

# Key that will be added to the state dictionary for maintenance reasons.
_INTERNAL_KEY = '_ckpt_internal'

def save_checkpoint(ckpt_dict, file_path, performance_score, train_iter=None,
                    max_ckpts_to_keep=5, keep_cktp_every=2, timestamp=None):
    """Save checkpoint to file.

    Example:
        .. code-block:: python

            save_checkpoint({
                'state_dict': net.state_dict(),
                'train_iter': curr_iteration
            }, 'ckpts/my_net', current_test_accuracy)

    Args:
        ckpt_dict: A dict with mostly arbitrary content. Though, most important,
                   it needs to include the state dict and should also include
                   the current training iteration.
        file_path: Where to store the checkpoint. Note, the filepath should
                   not change. Instead, 'train_iter' should be provided,
                   such that this method can handle the filenames by itself.
        performance_score: A score that expresses the performance of the
                           current network state, e.g., accuracy for a
                           classification task. This score is used to
                           maintain the list of kept checkpoints during
                           training.
        train_iter (optional): If given, it will be added to the filename.
            Otherwise, existing checkpoints are simply overwritten.
        max_ckpts_to_keep: The maximum number of checkpoints to
            keep. This will use the performance score to determine the n-1
            checkoints not to be deleted (where n is the number of
            checkpoints to keep). The current checkpoint will always be saved.
        keep_cktp_every: If this option is not :code:`None`,
            then every n hours one checkpoint will be permanently saved, i.e.,
            this checkpoint will not be maintained by 'max_ckpts_to_keep'
            anymore. The checkpoint to be kept will be the best one from the
            time window that spans the last n hours.
        timestamp (optional): The timestamp of this checkpoint. If not given,
            a current timestamp will be used. This option is useful when one
            aims to synchronize checkpoint savings from multiple networks.
    """
    if timestamp is None:
        ts = time.time() # timestamp
    else:
        ts = timestamp

    assert('state_dict' in ckpt_dict.keys())
    # We need to store internal (checkpoint maintenance related) information in
    # each checkpoint.
    internal_key = _INTERNAL_KEY
    assert(internal_key not in ckpt_dict.keys())
    ckpt_dict[internal_key] = dict()
    ckpt_dict[internal_key]['permanent'] = False
    ckpt_dict[internal_key]['score'] = performance_score
    ckpt_dict[internal_key]['ts']= ts

    # FIXME We currently don't care about file extensions.
    dname, fname = os.path.split(file_path)
    # Where do we store meta data, needed for maintenance.
    meta_fn = ('.' if not fname.startswith('.') else '') + fname + '_meta'
    meta_fn = os.path.join(dname, meta_fn)

    if not os.path.exists(dname):
        os.makedirs(dname)

    # Needed for option 'keep_cktp_every'. When was the first ckpt stored?
    if not os.path.exists(meta_fn):
        with open(meta_fn, 'w') as f:
            json.dump({'init_ts': ts}, f)
        init_ts = ts
    else:
        with open(meta_fn) as f:
            meta_dict = json.load(f)
        init_ts = meta_dict['init_ts']

    hrs_passed = (ts - init_ts) / (60 * 60)

    ### Iterate all existing checkpoints to determine which we remove.
    ckpt_fns = [os.path.join(dname, f) for f in os.listdir(dname) if
                os.path.isfile(os.path.join(dname, f)) and
                f.startswith(fname)]

    kept_ckpts = []
    permanent_ckpts = []

    for fn in ckpt_fns:
        # FIXME loading all checkpoints is expensive.
        ckpt = torch.load(fn)

        if not internal_key in ckpt:
            continue

        if ckpt[internal_key]['permanent']:
            permanent_ckpts.append((fn, ckpt[internal_key]['ts']))
        else:
            kept_ckpts.append((fn, ckpt[internal_key]['ts'],
                               ckpt[internal_key]['score']))

    ## Decide, whether a new permanent checkpoint should be saved.
    if keep_cktp_every is not None and hrs_passed >= keep_cktp_every:
        perm_ckpt_needed = True

        num_wins = hrs_passed // keep_cktp_every
        win_start = (num_wins-1) * keep_cktp_every

        # Check whether a permanent checkpoint for the current window already
        # exists.
        if len(permanent_ckpts) > 0:
            permanent_ckpts.sort(key=lambda tup: tup[1], reverse=True)

            ts_last_perm = permanent_ckpts[0][1]
            if ((ts_last_perm - init_ts) / (60 * 60)) >= win_start:
                perm_ckpt_needed = False

        if perm_ckpt_needed:
            # Choose the checkpoint with the best score in the current window
            # as next permanent checkpoint.
            kept_ckpts.sort(key=lambda tup: tup[1], reverse=True)
            max_score = -1
            max_ind = -1

            for i, tup in enumerate(kept_ckpts):
                if ((tup[1] - init_ts) / (60 * 60)) < win_start:
                    break

                if max_ind == -1 or max_score < tup[2]:
                    max_ind = i
                    max_score = tup[2]

            if max_ind != -1 and max_score > performance_score:
                # Transform an existing checkpoint into a permanent one.
                ckpt_tup = kept_ckpts[max_ind]
                # Important, we need to remove this item from the kept_ckpts,
                # as this list is used in the next step to determine which
                # checkpoints are removed.
                del kept_ckpts[max_ind]
                print('Checkpoint %s will be kept permanently.' % ckpt_tup[0])

                # FIXME: We might need the device here as in the load method.
                ckpt = torch.load(ckpt_tup[0])
                ckpt[internal_key]['permanent'] = True
                torch.save(ckpt, ckpt_tup[0])

            else:
                print('New checkpoint will be kept permanently.')
                ckpt_dict[internal_key]['permanent'] = True

    ## Decide, whether a checkpoint has to be deleted.
    if len(kept_ckpts) >= max_ckpts_to_keep:
        kept_ckpts.sort(key=lambda tup: tup[2])

        for i in range(len(kept_ckpts) - (max_ckpts_to_keep-1)):
            fn = kept_ckpts[i][0]
            print('Deleting old checkpoint: %s.' % fn)
            os.remove(fn)

    ### Save new checkpoint.
    if train_iter is not None:
        file_path += '_%d' % train_iter

    torch.save(ckpt_dict, file_path)
    print('Checkpoint saved to %s' % file_path)

def load_checkpoint(ckpt_path, net, device=None, ret_performance_score=False):
    """Load a checkpoint from file.

    Args:
        ckpt_path: Path to checkpoint.
        net: The network, that should load the state dict saved in this
             checkpoint.
        device (optional): The device currently used by the model. Can help to
                           speed up loading the checkpoint.
        ret_performance_score: If True, the score associated with this
            checkpoint will be returned as well. See argument
            "performance_score" of method "save_ckecpoint".

    Returns:
        The loaded checkpoint. Note, the state_dict is already applied to the
        network. However, there might be other important dict elements.
    """
    # See here for details on how to load the checkpoint:
    # https://blog.floydhub.com/checkpointing-tutorial-for-tensorflow-keras-and-pytorch/
    if device is not None and device.type == 'cuda':
        ckpt = torch.load(ckpt_path)
    else:
        # Load GPU model on CPU
        ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)

    net.load_state_dict(ckpt['state_dict'])

    if ret_performance_score:
        score = ckpt[_INTERNAL_KEY]['score']

    # That key was added for maintenance reasons in the method save_checkpoint.
    if _INTERNAL_KEY in ckpt:
        del ckpt[_INTERNAL_KEY]

    if ret_performance_score:
        return ckpt, score

    return ckpt

def make_ckpt_list(file_path):
    """Creates a file that lists all checkpoints together with there scores,
    such that one can easily find the checkpoint associated with the maximum
    score.

    Args:
        file_path: See method :func:`save_checkpoints`.
    """
    internal_key = _INTERNAL_KEY

    dname, fname = os.path.split(file_path)

    assert(os.path.exists(dname))

    ckpt_fns = [(f, os.path.join(dname, f)) for f in os.listdir(dname) if
                os.path.isfile(os.path.join(dname, f)) and
                f.startswith(fname)]

    ckpts = []

    for fn, fpath in ckpt_fns:
        ckpt = torch.load(fpath)

        if not internal_key in ckpt:
            continue

        score = ckpt[internal_key]['score']

        ckpts.append((fn, score))

    ckpts.sort(key=lambda tup: tup[1], reverse=True)

    with open(os.path.join(dname, 'score_list_' + fname + '.txt'), 'w') as f:
        for tup in ckpts:
            f.write('%s, %f\n' % (tup[0], tup[1]))

def get_best_ckpt_path(file_path):
    """Returns the path to the checkpoint with the highest score.

    Args:
        file_path: See method :func:`save_checkpoints`.
    """
    dname, fname = os.path.split(file_path)
    assert(os.path.exists(dname))

    # See method make_ckpt_list.
    ckpt_list_fn = os.path.join(dname, 'score_list_' + fname + '.txt')
    if os.path.exists(ckpt_list_fn):
        with open(ckpt_list_fn, 'r') as f:
            # Get first word from file. Note, the filename ends with a comma.
            best_ckpt_fname = f.readline().split(None, 1)[0][:-1]

        return os.path.join(dname, best_ckpt_fname)

    # Go through each checkpoint and evaluate the score achieved.
    ckpt_fns = [(f, os.path.join(dname, f)) for f in os.listdir(dname) if
                os.path.isfile(os.path.join(dname, f)) and
                f.startswith(fname)]

    best_ckpt_path = None
    best_score = -1

    for fn, fpath in ckpt_fns:
        ckpt = torch.load(fpath)

        if not _INTERNAL_KEY in ckpt:
            continue

        score = ckpt[_INTERNAL_KEY]['score']
        if score > best_score:
            best_score = score
            best_ckpt_path = fpath

    return best_ckpt_path

if __name__ == '__main__':
    pass


