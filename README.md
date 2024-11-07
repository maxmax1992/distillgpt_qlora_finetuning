## Document classification with distillbert Q-Lora
Model is using Lora to finetune DistillGPT to classify documents (Q-Lora is not supported by M1 arch macs unfortunately :(, issue [here](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/485#issuecomment-1636948744))

### Running instructions
install dependencies: 
`poetry shell`
`poetry install`
run command:
`python distillgpt_finetuning/train.py --batch_size=32`
to run in dry mode:
`python distillgpt_finetuning/train.py --batch_size=5 --dry_run`
the csv will be created under the root and the models saved to ./models/ep_validation_loss.pth, choose model with lowest loss to use


#### Things to improve the model
1. Optimize the forward pass to be more paralellizable and perform batch computation in single pass.
2. Also, other features based on text inputs, like intervals between dates and better text cleaning etc.
3. Better embedding of named entities, e.g. names should be categorical values defined by the id and embedded with nn.Embedding
4. Create a dockerfile, compose.yaml for linux based system for faster cloud training.
5. More experimentation e.g. use frozen model without explicit fine-tuning, and provide metadata for the rows as well e.g. speakers: .... .
6. Try different models, e.g. frozen Bert