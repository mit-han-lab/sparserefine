DOCKER_NAME=$(basename $(pwd))

docker build . -t $DOCKER_NAME
wandb docker-run \
    --gpus all \
    --user $(id -u):$(id -g) \
    --volume $(pwd):/workspace \
    --volume /dataset:/dataset \
    --env PYTHONPATH=. \
    --env WANDB_CACHE_DIR=/tmp/wandb/cache/ \
    --env WANDB_CONFIG_DIR=/tmp/wandb/config/ \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --interactive --tty --rm \
    $DOCKER_NAME "$@"
