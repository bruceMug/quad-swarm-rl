# Keep wandb off the AFS home volume.
#
# AFS home (/afs/andrew.cmu.edu/usr6/bmugizi) sits at ~90 % of its 2 GB quota, and AFS
# surfaces "quota exceeded" as EACCES. wandb writes its cache, run data and artifact staging
# under ~/.cache/wandb and ~/.local/share/wandb by default, so a long rollout finishes, writes
# its .npz/.json, and then fails at the logging step with:
#
#   PermissionError: [Errno 13] ... /.local/share/wandb/artifacts/staging/tmpXXXX
#
# The rollout itself is unaffected -- collect.py writes results before logging and treats a
# logging failure as non-fatal -- but the run never reaches wandb. Pointing every wandb path
# at /scratch fixes it for good.
#
#     source scripts/byz_wandb_env.sh
#
# Add it to ~/.bashrc to make it permanent.
export WANDB_DIR=/scratch/bmugizi/wandb_home
export WANDB_CACHE_DIR=/scratch/bmugizi/wandb_home/cache
export WANDB_DATA_DIR=/scratch/bmugizi/wandb_home/data
export WANDB_ARTIFACT_DIR=/scratch/bmugizi/wandb_home/artifacts
export WANDB_CONFIG_DIR=/scratch/bmugizi/wandb_home/config
# wandb's Go core also honours XDG paths for its debug logs.
export XDG_CACHE_HOME=/scratch/bmugizi/wandb_home/cache
export XDG_DATA_HOME=/scratch/bmugizi/wandb_home/data

# Credentials, too. wandb reads ~/.netrc, which lives on AFS -- and AFS access needs a valid
# token. Tokens expire (~25 h); a rollout started in a long-lived tmux session can outlive its
# token and then fail at the very end with
#
#   PermissionError: [Errno 13] Permission denied: '/afs/.../.netrc'
#
# even though nothing is wrong with the file. wandb honours $NETRC (wbauth/wbnetrc.py), so
# point it at the identical copy on /scratch, which needs no token.
export NETRC=/scratch/bmugizi/.netrc
