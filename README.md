# llama-3-2-vision-finetune

This repository contains step by step guide for for the AIM 368 Chalk Talk at AWS re:Invent 2024

This guide contains 4 part:
1. Training cluster setup
2. Finetune Llama 3.2 vision model
3. Deploy the fine tuned model
4. Automate Web tasks with SeeAct framework and finetuned model

## Training Infrastructure Setup
https://aws.amazon.com/blogs/machine-learning/scale-llms-with-pytorch-2-0-fsdp-on-amazon-eks-part-2/

https://github.com/aws-samples/aws-do-eks/tree/main/wd/conf/terraform/eks-p5


## Finetune Llama 3.2 11B Vision model
Core training arguments is as below using llama receipe:
```
  torchrun --nnodes 1 --nproc_per_node 8  recipes/quickstart/finetuning/finetuning.py --enable_fsdp --lr 1e-5  --num_epochs 5 --batch_size_training 2 --model_name meta-llama/Llama-3.2-11B-Vision-Instruct --dist_checkpoint_root_folder ./finetuned_model --dist_checkpoint_folder fine-tuned  --use_fast_kernels --dataset "custom_dataset" --custom_dataset.test_split "test" --custom_dataset.file "recipes/quickstart/finetuning/datasets/mind2web_dataset.py"  --run_validation True --batching_strategy padding
```

We will use PyTorch FSDP for training on EKS cluster, follow this guide to setup EKS cluster and deplpoy the training job

https://github.com/roywei/aws-do-eks/tree/llama-3-2-vision/Container-Root/eks/deployment/distributed-training/pytorch/pytorchjob/fsdp

It's been modified to extend AWS DLC for llama 3.2 finetuning

Remember to uploaded finetuned model to s3 after job is done.

## Deployment

We provide 3 deployment options based on your needs
### Deploy on AWS Bedrock

You can import custom model on AWS Bedrock Console by following this guide https://aws.amazon.com/bedrock/custom-model-import/
Simply click import model and select the S3 bucket with the finetune model, you will be able to run Bedrock inference with the same API as default llama 3.2 vision model.

### Deploy on AWS SageMaker with Stateful Inference
Follow this guide to deploy on AWS SageMaker with stateful infernece capabilities: https://github.com/aws-samples/sagemaker-genai-hosting-examples/tree/main/Llama3/llama3-11b-vision/stateful


### Deploy on EC2 using SGLang
```
docker run --gpus all \
    --shm-size 32g \
    -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=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" \
    --ipc=host \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server --model-path meta-llama/Llama-3.2-11B-Vision-Instruct --host 0.0.0.0 --port 30000 --chat-template llama_3_vision --api-key myapikeysglang
```

## SeeAct Setup

Clone and deploy SeeAct framework on your local laptop to let LLM automate web tasks.
https://github.com/roywei/SeeAct/tree/sglang