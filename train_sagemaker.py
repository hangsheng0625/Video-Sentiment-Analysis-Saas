from sagemaker.pytorch import PyTorch
from sagemaker.debugger import TensorBoardOutputConfig
import sagemaker

def start_training():
    # Use your existing bucket
    session = sagemaker.Session(default_bucket='video-sentiment-analysis-model')
    
    tensorboard_config = TensorBoardOutputConfig(
        s3_output_path='s3://video-sentiment-analysis-model/tensorboard',
        container_local_output_path='/opt/ml/output/tensorboard'
    )

    estimator = PyTorch(
        entry_point='train.py',
        source_dir='training',
        role='arn:aws:iam::843976229045:role/video-sentiment-analysis-execution-role',
        instance_count=1,
        instance_type='ml.g5.xlarge',
        framework_version='2.5.1',
        py_version='py311',
        hyperparameters={
            'epochs': 25,
            'batch_size': 32
        },
        tensorboard_output_config=tensorboard_config,
        sagemaker_session=session,  # Use custom session
        output_path='s3://video-sentiment-analysis-model/models/',  # Specify output path
        dependencies=['training/requirements.txt']  

    )
    
    estimator.fit({
        'training': 's3://video-sentiment-analysis-model/dataset/train/',
        'validation': 's3://video-sentiment-analysis-model/dataset/dev/',
        'test': 's3://video-sentiment-analysis-model/dataset/test/'
    }, wait = True, logs=True) 

if __name__ == '__main__':
    start_training()