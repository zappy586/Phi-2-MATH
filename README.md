# Phi-2-MATH

![Designer (2)](https://github.com/zappy586/Phi-2-MATH/assets/89218647/3fc03bdd-ccdd-417c-8124-3cae95407616)

## This is a colab notebook for finetuning Microsoft's Phi-2-3B LLM for solving mathematical word problems using QLoRA, Uploading adapters to 🤗 Hub, Merging the adapters and then uploading it on 🤗 repo. The notebook also contains code for inferencing it directly from my repo.

* Link to my repo: https://huggingface.co/ZappY-AI/phi2-math-orca
* Link to the dataset: https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k
* Base model: https://huggingface.co/microsoft/phi-2

* The model was loaded and trained in 4-bit quantization and float16 tensors for the sake of efficiency.

* The model was chosen because of its small size and less trainable parameters and extremely good performance as it can outperform models that are 5x-10x times bigger than itself. Take a look at these eval metrics:

![image](https://github.com/zappy586/Phi-2-MATH/assets/89218647/316e1cdc-8900-4cad-a58b-b063a26f3bb7)

* The model was trained for 500 steps on a T4 colab pro GPU (16 GB VRAM) for about 2.5 hours on a subset(20%) of the original dataset using TRL's SFT Trainer.
* The training loss obtained at the final step was <ins>0.556700
* The following is the PEFT Config for this training notebook:
  ```
  peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules= ["Wqkv", "out_proj"])
  ```
  * The following are the training metrics:
    ![image](https://github.com/zappy586/Phi-2-MATH/assets/89218647/aa036b58-864f-40eb-891a-a798b72fae44)

  
