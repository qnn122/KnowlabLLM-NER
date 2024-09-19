Inherited from Stanford's [Alpaca](https://github.com/tatsu-lab/stanford_alpaca)

## Setup
### Generate data for instruction finetuning
Run `generate_instruction_bioner.py` to produce NER finetuning dataset `./data/instruction_bioner.json`

### Finetune LLM for NER task
Run `train.py` (see `train.sh`), model produced will be store in `models/`