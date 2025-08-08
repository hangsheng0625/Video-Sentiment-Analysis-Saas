from sagemaker.pytorch import PyTorch
from sagemaker.debugger import TensorBoardOutputConfig

def start_training():
    tensorboard_config = TensorBoardOutputConfig(
        s3_output_path='bucket-name/tensorboard',
        container_local_output_path='/opt/ml/output/tensorboard'
    )

    estimator = PyTorch(
        entry_point='train.py',
        source_dir='training',
        role='my-new-role',
        instance_count=1,
        instance_type='ml.g5.xlarge',
        framework_version='2.5.1',
        py_version='py311',
        hyperparameters={
            'epochs': 25,
            'batch_size': 32        },
        tensorboard_output_config=tensorboard_config
    )

    # Start the training job
    estimator.fit({
        'training': 's3://bucket-name/dataset/train/',
        'validation': 's3://bucket-name/dataset/val/',
        'test': 's3://bucket-name/dataset/test/'
    })

if __name__ == '__main__':
    start_training()
    