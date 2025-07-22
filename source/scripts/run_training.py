import os


def main():
    # Параметры для запуска обучения
    params = {
        "model": "efficientnet_b0",
        "data_dir": "D:/Users/Legion/datasets/imagenet/train",
        "dataset": "imagenet",
        "batch_size": 16,
        "epochs": 100,
        "output": "D:/Users/Legion/Polytech/DeepLearning/lab_1/pytorch-image-models-main/models/efficientnet_v2_1",
        "sched": "cosine",
        "opt": "adamw",
        "lr": 0.001,
        "weight_decay": 1e-4,
        "device": "cuda",
        "workers": 8,
        "log_interval": 50,
        "amp": True  # Включение AMP (смешанная точность)
    }

    # Проверка наличия флага AMP
    amp_flag = "--amp" if params['amp'] else ""

    # Формирование команды для запуска
    command = (
        f"python train.py "
        f"--model {params['model']} "
        f"--data-dir {params['data_dir']} "
        f"--dataset {params['dataset']} "
        f"--batch-size {params['batch_size']} "
        f"--epochs {params['epochs']} "
        f"--output {params['output']} "
        f"--sched {params['sched']} "
        f"--opt {params['opt']} "
        f"--lr {params['lr']} "
        f"--weight-decay {params['weight_decay']} "
        f"--device {params['device']} "
        f"--workers {params['workers']} "
        f"--log-interval {params['log_interval']} "
        f"{amp_flag}"
    )

    # Вывод команды для отладки (опционально)
    print(f"Running command: {command}")

    # Запуск команды
    os.system(command)


if __name__ == '__main__':
    main()
