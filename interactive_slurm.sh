#!/usr/bin/env bash

srun --job-name Terminal --partition dios --gres=gpu:1 --pty /bin/bash
